"""
Agent classes for the ACE framework.

This module implements the three core agents: Generator, Reflector, and Curator.
These agents work together to create an adaptive context engineering system.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import logging
from .llm_client import LLMClient, Message
from .context import Bullet, ContextPlaybook
from .utils import SemanticSimilarity


class GeneratorAgent:
    """The Generator agent produces reasoning trajectories.

    The Generator uses the current context playbook to generate responses
    to queries. It also provides feedback on which bullets were helpful
    or harmful during generation.

    Attributes:
        llm_client: LLM client instance for API calls.
        logger: Logger instance for agent operations.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the Generator agent.

        Args:
            llm_client: LLM client instance for making API calls.
        """
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__ + ".GeneratorAgent")

    def generate(
        self, query: str, playbook: ContextPlaybook, temperature: float = 0.7, **kwargs: Any
    ) -> Tuple[str, Dict[str, List[str]]]:
        """Generate a response using the current playbook.

        Args:
            query: The user query to process.
            playbook: The current context playbook.
            temperature: Temperature for sampling. Defaults to 0.7.
            **kwargs: Additional generation parameters.

        Returns:
            Tuple containing:
                - response (str): The generated response text
                - bullet_feedback (Dict[str, List[str]]): Dictionary with 'helpful' and 'harmful' bullet IDs

        Raises:
            Exception: If generation fails due to API errors.
        """
        self.logger.info(f"Generating response for query: {query[:100]}...")

        # Build prompt with context injection
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, playbook)

        try:
            # Call LLM API
            response = self.llm_client.simple_completion(
                prompt=user_prompt, system_prompt=system_prompt, temperature=temperature, **kwargs
            )

            # Extract bullet feedback from response
            bullet_feedback = self._extract_bullet_feedback(response, playbook)

            self.logger.info(
                f"Generated response with {len(bullet_feedback['helpful'])} helpful bullets"
            )

            return response, bullet_feedback

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise

    def _build_system_prompt(self) -> str:
        """Build the system prompt for generation.

        Returns:
            System prompt string with instructions for using the playbook.
        """
        return """You are an expert assistant that uses a context playbook to provide accurate responses.

When answering queries:
1. Carefully review the context playbook provided
2. Reference specific bullet IDs (e.g., [ctx-001]) when using strategies or rules from the playbook
3. Provide detailed, well-reasoned responses
4. If the playbook contains relevant guidance, follow it closely
5. Be explicit about which bullets you found helpful or unhelpful

Your response should be clear, accurate, and demonstrate how you applied the playbook context."""

    def _build_user_prompt(self, query: str, playbook: ContextPlaybook) -> str:
        """Build the user prompt with context injection.

        Args:
            query: The user query.
            playbook: The context playbook to inject.

        Returns:
            User prompt string with playbook context and query.
        """
        if len(playbook) == 0:
            return f"Query: {query}\n\nPlease provide a detailed response."

        context_str = playbook.to_prompt(include_scores=False)

        prompt = f"""{context_str}

---

Query: {query}

Please provide a detailed response, referencing relevant bullet IDs from the playbook above when applicable."""

        return prompt

    def _extract_bullet_feedback(
        self, response: str, playbook: ContextPlaybook
    ) -> Dict[str, List[str]]:
        """Extract bullet feedback from the generated response.

        This method looks for bullet ID references in the response to determine
        which bullets were used (considered helpful).

        Args:
            response: The generated response text.
            playbook: The context playbook used for generation.

        Returns:
            Dictionary with keys 'helpful' and 'harmful', each containing a list of bullet IDs.
        """
        feedback: Dict[str, List[str]] = {"helpful": [], "harmful": []}

        # Extract all bullet ID references (e.g., [ctx-001], ctx-abc123)
        bullet_pattern = r"\[?(ctx-[a-f0-9]{8})\]?"
        mentioned_bullets = re.findall(bullet_pattern, response, re.IGNORECASE)

        # Get all valid bullet IDs from playbook
        valid_bullet_ids = {bullet.id for bullet in playbook.bullets}

        # Mark mentioned bullets as helpful
        for bullet_id in set(mentioned_bullets):
            if bullet_id in valid_bullet_ids:
                feedback["helpful"].append(bullet_id)

        return feedback


class ReflectorAgent:
    """The Reflector agent analyzes trajectories and extracts insights.

    The Reflector examines execution results, compares them with ground truth,
    diagnoses errors, and extracts key insights for context updates.

    Attributes:
        llm_client: LLM client instance for API calls.
        logger: Logger instance for agent operations.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the Reflector agent.

        Args:
            llm_client: LLM client instance for making API calls.
        """
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__ + ".ReflectorAgent")

    def reflect(
        self,
        query: str,
        trajectory: str,
        ground_truth: Optional[str] = None,
        execution_result: Optional[Dict[str, Any]] = None,
        playbook: Optional[ContextPlaybook] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Analyze a trajectory and extract insights.

        Args:
            query: The original query.
            trajectory: The generated trajectory to analyze.
            ground_truth: The ground truth answer for comparison. Defaults to None.
            execution_result: Execution feedback from running the trajectory. Defaults to None.
            playbook: The current playbook for bullet tagging. Defaults to None.
            **kwargs: Additional parameters for LLM calls.

        Returns:
            Dictionary containing insights with keys:
                - errors: List of identified errors
                - root_cause: Root cause analysis
                - insights: List of actionable insights
                - bullet_tags: Dictionary mapping bullet IDs to tags (HELPFUL/HARMFUL/NEUTRAL)
                - recommendations: List of recommendations
                - raw_response: Raw LLM response

        Raises:
            Exception: If reflection fails due to API errors.
        """
        self.logger.info("Reflecting on trajectory...")

        # Build reflection prompt
        prompt = self._build_reflection_prompt(
            query=query,
            trajectory=trajectory,
            ground_truth=ground_truth,
            execution_result=execution_result,
            playbook=playbook,
        )

        try:
            # Call LLM for reflection
            reflection_response = self.llm_client.simple_completion(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more focused analysis
                **kwargs,
            )

            # Parse insights from response
            insights = self._parse_reflection_response(reflection_response)

            self.logger.info(
                f"Reflection complete: {len(insights.get('insights', []))} insights extracted"
            )

            return insights

        except Exception as e:
            self.logger.error(f"Reflection failed: {e}")
            raise

    def _build_reflection_prompt(
        self,
        query: str,
        trajectory: str,
        ground_truth: Optional[str],
        execution_result: Optional[Dict[str, Any]],
        playbook: Optional[ContextPlaybook],
    ) -> str:
        """Build the reflection prompt.

        Args:
            query: The original query.
            trajectory: The generated trajectory.
            ground_truth: Ground truth answer for comparison. Defaults to None.
            execution_result: Execution feedback. Defaults to None.
            playbook: Context playbook for bullet analysis. Defaults to None.

        Returns:
            Formatted reflection prompt string.
        """
        prompt_parts = [
            "You are an expert analyzer tasked with reflecting on a reasoning trajectory.",
            "",
            f"**Original Query:**",
            query,
            "",
            f"**Generated Trajectory:**",
            trajectory,
            "",
        ]

        # Add ground truth if available
        if ground_truth:
            prompt_parts.extend([f"**Ground Truth:**", ground_truth, ""])

        # Add execution result if available
        if execution_result:
            prompt_parts.extend([f"**Execution Result:**", str(execution_result), ""])

        # Add playbook context if available
        if playbook and len(playbook) > 0:
            prompt_parts.extend(
                [f"**Context Playbook Used:**", playbook.to_prompt(include_scores=False), ""]
            )

        # Add analysis instructions
        prompt_parts.extend(
            [
                "**Your Task:**",
                "Please provide a thorough analysis of this trajectory. Include:",
                "",
                "1. **ERROR_ANALYSIS**: Identify any errors, inaccuracies, or issues in the reasoning",
                "2. **ROOT_CAUSE**: Determine the root cause of any errors",
                "3. **KEY_INSIGHTS**: Extract actionable insights that could improve future responses",
                "4. **BULLET_TAGS**: If a playbook was used, tag each bullet as:",
                "   - HELPFUL: Led to correct reasoning",
                "   - HARMFUL: Led to incorrect reasoning  ",
                "   - NEUTRAL: Not relevant or unclear impact",
                "5. **RECOMMENDATIONS**: Suggest specific improvements or new strategies",
                "",
                "Format your response with clear section headers (ERROR_ANALYSIS, ROOT_CAUSE, etc.).",
                "Be specific and actionable in your analysis.",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_reflection_response(self, response: str) -> Dict[str, Any]:
        """Parse the reflection response into structured insights.

        Args:
            response: The LLM's reflection response text.

        Returns:
            Structured insights dictionary with keys: errors, root_cause, insights,
            bullet_tags, recommendations, and raw_response.
        """
        insights: Dict[str, Any] = {
            "errors": [],
            "root_cause": "",
            "insights": [],
            "bullet_tags": {},
            "recommendations": [],
        }

        # Extract sections using regex
        sections = {
            "ERROR_ANALYSIS": r"ERROR_ANALYSIS[:\s]+(.*?)(?=ROOT_CAUSE|KEY_INSIGHTS|BULLET_TAGS|RECOMMENDATIONS|$)",
            "ROOT_CAUSE": r"ROOT_CAUSE[:\s]+(.*?)(?=KEY_INSIGHTS|BULLET_TAGS|RECOMMENDATIONS|$)",
            "KEY_INSIGHTS": r"KEY_INSIGHTS[:\s]+(.*?)(?=BULLET_TAGS|RECOMMENDATIONS|$)",
            "BULLET_TAGS": r"BULLET_TAGS[:\s]+(.*?)(?=RECOMMENDATIONS|$)",
            "RECOMMENDATIONS": r"RECOMMENDATIONS[:\s]+(.*?)$",
        }

        for section_name, pattern in sections.items():
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()

                if section_name == "ERROR_ANALYSIS":
                    # Extract error list
                    errors = [
                        line.strip("- ").strip() for line in content.split("\n") if line.strip()
                    ]
                    insights["errors"] = errors

                elif section_name == "ROOT_CAUSE":
                    insights["root_cause"] = content

                elif section_name == "KEY_INSIGHTS":
                    # Extract insights list
                    key_insights = [
                        line.strip("- ").strip() for line in content.split("\n") if line.strip()
                    ]
                    insights["insights"] = key_insights

                elif section_name == "BULLET_TAGS":
                    # Extract bullet tags (ctx-xxxxx: HELPFUL/HARMFUL/NEUTRAL)
                    bullet_pattern = r"(ctx-[a-f0-9]{8})[:\s]+(HELPFUL|HARMFUL|NEUTRAL)"
                    tags = re.findall(bullet_pattern, content, re.IGNORECASE)
                    for bullet_id, tag in tags:
                        insights["bullet_tags"][bullet_id] = tag.upper()

                elif section_name == "RECOMMENDATIONS":
                    # Extract recommendations
                    recs = [
                        line.strip("- ").strip() for line in content.split("\n") if line.strip()
                    ]
                    insights["recommendations"] = recs

        # Store raw response for debugging
        insights["raw_response"] = response

        return insights

    def iterative_refine(
        self, initial_insights: Dict[str, Any], max_iterations: int = 3, **kwargs: Any
    ) -> Dict[str, Any]:
        """Iteratively refine insights through multiple reflection rounds.

        Args:
            initial_insights: The initial insights to refine.
            max_iterations: Maximum number of refinement iterations. Defaults to 3.
            **kwargs: Additional parameters for LLM calls.

        Returns:
            Refined insights dictionary with improved quality and specificity.
        """
        self.logger.info(f"Starting iterative refinement: {max_iterations} iterations")

        current_insights = initial_insights

        for iteration in range(max_iterations):
            self.logger.info(f"Refinement iteration {iteration + 1}/{max_iterations}")

            # Build refinement prompt
            prompt = self._build_refinement_prompt(current_insights, iteration)

            try:
                # Get refined insights
                refined_response = self.llm_client.simple_completion(
                    prompt=prompt, temperature=0.3, **kwargs
                )

                # Parse refined insights
                current_insights = self._parse_reflection_response(refined_response)

            except Exception as e:
                self.logger.error(f"Refinement iteration {iteration + 1} failed: {e}")
                break

        self.logger.info("Iterative refinement complete")
        return current_insights

    def _build_refinement_prompt(self, current_insights: Dict[str, Any], iteration: int) -> str:
        """Build a prompt for refining existing insights.

        Args:
            current_insights: Current insights to refine.
            iteration: Current iteration number (0-indexed).

        Returns:
            Refinement prompt string asking for improved insights.
        """
        prompt = f"""You are refining an analysis. This is refinement iteration {iteration + 1}.

**Current Insights:**

ERROR_ANALYSIS:
{chr(10).join('- ' + e for e in current_insights.get('errors', []))}

ROOT_CAUSE:
{current_insights.get('root_cause', 'Not identified')}

KEY_INSIGHTS:
{chr(10).join('- ' + i for i in current_insights.get('insights', []))}

RECOMMENDATIONS:
{chr(10).join('- ' + r for r in current_insights.get('recommendations', []))}

**Your Task:**
Review and refine these insights:
1. Remove any vague or unhelpful points
2. Make insights more specific and actionable
3. Consolidate similar points
4. Add any missing critical insights

Provide the refined analysis using the same format (ERROR_ANALYSIS, ROOT_CAUSE, KEY_INSIGHTS, RECOMMENDATIONS)."""

        return prompt


class CuratorAgent:
    """The Curator agent manages playbook updates.

    The Curator takes insights from the Reflector and generates delta
    context updates. It handles adding, updating, and removing bullets,
    as well as deduplication.

    Attributes:
        llm_client: LLM client instance for API calls.
        logger: Logger instance for agent operations.
        semantic_similarity: Semantic similarity calculator for deduplication.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the Curator agent.

        Args:
            llm_client: LLM client instance for making API calls.
        """
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__ + ".CuratorAgent")
        self.semantic_similarity = SemanticSimilarity()

    def curate(
        self, insights: Dict[str, Any], playbook: ContextPlaybook, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Generate delta context updates from insights.

        Args:
            insights: Insights from the Reflector agent.
            playbook: The current context playbook.
            **kwargs: Additional parameters for LLM calls.

        Returns:
            List of delta operation dictionaries, each containing 'operation' (ADD/UPDATE/REMOVE),
            and relevant fields like 'content', 'section', 'bullet_id', and 'reason'.

        Raises:
            Exception: If curation fails due to API errors.
        """
        self.logger.info("Curating playbook updates from insights...")

        # Build curation prompt
        prompt = self._build_curation_prompt(insights, playbook)

        try:
            # Call LLM for curation
            curation_response = self.llm_client.simple_completion(
                prompt=prompt, temperature=0.3, **kwargs  # Lower temperature for structured output
            )

            # Parse delta operations
            deltas = self._parse_delta_operations(curation_response)

            # Add automatic REMOVE operations for harmful bullets
            deltas.extend(self._generate_harmful_removals(insights))

            self.logger.info(f"Generated {len(deltas)} delta operations")

            return deltas

        except Exception as e:
            self.logger.error(f"Curation failed: {e}")
            raise

    def _build_curation_prompt(self, insights: Dict[str, Any], playbook: ContextPlaybook) -> str:
        """Build the curation prompt.

        Args:
            insights: Insights from reflection.
            playbook: Current context playbook.

        Returns:
            Formatted curation prompt string with instructions for generating delta operations.
        """
        prompt_parts = [
            "You are a context curator responsible for maintaining a high-quality playbook.",
            "",
            "**Current Playbook:**",
        ]

        if len(playbook) > 0:
            prompt_parts.append(playbook.to_prompt(include_scores=True))
        else:
            prompt_parts.append("(Empty playbook)")

        prompt_parts.extend(["", "**Insights from Reflection:**", ""])

        # Add insights
        if insights.get("insights"):
            prompt_parts.append("Key Insights:")
            for insight in insights["insights"]:
                prompt_parts.append(f"- {insight}")
            prompt_parts.append("")

        # Add recommendations
        if insights.get("recommendations"):
            prompt_parts.append("Recommendations:")
            for rec in insights["recommendations"]:
                prompt_parts.append(f"- {rec}")
            prompt_parts.append("")

        # Add bullet tags
        if insights.get("bullet_tags"):
            prompt_parts.append("Bullet Tags:")
            for bullet_id, tag in insights["bullet_tags"].items():
                prompt_parts.append(f"- {bullet_id}: {tag}")
            prompt_parts.append("")

        # Add instructions
        prompt_parts.extend(
            [
                "**Your Task:**",
                "Generate delta operations to update the playbook based on these insights.",
                "",
                "For each operation, use this exact format:",
                "",
                "OPERATION: ADD",
                "CONTENT: <the bullet content>",
                "SECTION: <strategies_and_hard_rules | troubleshooting | general>",
                "REASON: <why this is being added>",
                "---",
                "",
                "OPERATION: UPDATE",
                "BULLET_ID: <ctx-xxxxxxxx>",
                "CONTENT: <updated content>",
                "REASON: <why this is being updated>",
                "---",
                "",
                "OPERATION: REMOVE",
                "BULLET_ID: <ctx-xxxxxxxx>",
                "REASON: <why this is being removed>",
                "---",
                "",
                "Guidelines:",
                "- ADD operations for new insights that should be preserved",
                "- UPDATE operations to refine existing bullets marked as HELPFUL",
                "- DO NOT generate REMOVE operations (these are handled automatically)",
                "- Be specific and actionable in content",
                "- Choose appropriate sections",
                "",
                "Generate delta operations:",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_delta_operations(self, response: str) -> List[Dict[str, Any]]:
        """Parse delta operations from LLM response.

        Args:
            response: LLM response text containing delta operations.

        Returns:
            List of parsed delta operation dictionaries.
        """
        deltas = []

        # Split by operation separator
        operations = response.split("---")

        for op_text in operations:
            op_text = op_text.strip()
            if not op_text:
                continue

            delta = self._parse_single_operation(op_text)
            if delta:
                deltas.append(delta)

        return deltas

    def _parse_single_operation(self, op_text: str) -> Optional[Dict[str, Any]]:
        """Parse a single delta operation.

        Args:
            op_text: Operation text to parse.

        Returns:
            Delta dictionary if parsing succeeds, None otherwise.
        """
        lines = op_text.split("\n")
        delta: Dict[str, Any] = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("OPERATION:"):
                operation = line.split(":", 1)[1].strip().upper()
                if operation in ["ADD", "UPDATE", "REMOVE"]:
                    delta["operation"] = operation

            elif line.startswith("BULLET_ID:"):
                delta["bullet_id"] = line.split(":", 1)[1].strip()

            elif line.startswith("CONTENT:"):
                delta["content"] = line.split(":", 1)[1].strip()

            elif line.startswith("SECTION:"):
                section = line.split(":", 1)[1].strip()
                if section in ["strategies_and_hard_rules", "troubleshooting", "general"]:
                    delta["section"] = section

            elif line.startswith("REASON:"):
                delta["reason"] = line.split(":", 1)[1].strip()

        # Validate delta
        if "operation" not in delta:
            return None

        operation = delta["operation"]

        if operation == "ADD":
            if "content" in delta and "section" in delta:
                return delta

        elif operation == "UPDATE":
            if "bullet_id" in delta and "content" in delta:
                return delta

        elif operation == "REMOVE":
            if "bullet_id" in delta:
                return delta

        return None

    def _generate_harmful_removals(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate REMOVE operations for harmful bullets.

        Args:
            insights: Insights dictionary containing bullet tags.

        Returns:
            List of REMOVE delta operations for bullets tagged as HARMFUL.
        """
        deltas = []
        bullet_tags = insights.get("bullet_tags", {})

        for bullet_id, tag in bullet_tags.items():
            if tag.upper() == "HARMFUL":
                deltas.append(
                    {
                        "operation": "REMOVE",
                        "bullet_id": bullet_id,
                        "reason": "Marked as harmful in reflection",
                    }
                )

        return deltas

    def apply_deltas(self, deltas: List[Dict[str, Any]], playbook: ContextPlaybook) -> None:
        """Apply delta operations to the playbook.

        Args:
            deltas: List of delta operation dictionaries.
            playbook: The playbook to update.
        """
        for delta in deltas:
            operation = delta.get("operation")

            if operation == "ADD":
                bullet = Bullet(
                    id=Bullet.generate_id(),
                    content=delta["content"],
                    section=delta.get("section", "general"),
                )
                playbook.add_bullet(bullet)

            elif operation == "UPDATE":
                playbook.update_bullet(
                    bullet_id=delta["bullet_id"],
                    content=delta.get("content"),
                    section=delta.get("section"),
                )

            elif operation == "REMOVE":
                playbook.remove_bullet(delta["bullet_id"])

    def deduplicate(self, playbook: ContextPlaybook, similarity_threshold: float = 0.9) -> int:
        """Remove duplicate bullets based on semantic similarity.

        Args:
            playbook: The playbook to deduplicate.
            similarity_threshold: Similarity threshold (0-1) for considering bullets as duplicates.
                Defaults to 0.9.

        Returns:
            Number of duplicate bullets removed.
        """
        if len(playbook) < 2:
            return 0

        self.logger.info(f"Deduplicating playbook with {len(playbook)} bullets...")

        # Extract bullet contents and IDs
        bullets = playbook.bullets
        contents = [bullet.content for bullet in bullets]

        # Find similar pairs
        similar_pairs = self.semantic_similarity.find_similar_pairs(
            contents, threshold=similarity_threshold
        )

        if not similar_pairs:
            self.logger.info("No duplicates found")
            return 0

        # Keep track of bullets to remove
        bullets_to_remove = set()

        for idx1, idx2, similarity in similar_pairs:
            bullet1 = bullets[idx1]
            bullet2 = bullets[idx2]

            # Skip if either bullet already marked for removal
            if bullet1.id in bullets_to_remove or bullet2.id in bullets_to_remove:
                continue

            # Decide which bullet to keep based on scores
            score1 = bullet1.get_score()
            score2 = bullet2.get_score()

            if score1 >= score2:
                # Keep bullet1, remove bullet2
                bullets_to_remove.add(bullet2.id)
                self.logger.debug(
                    f"Removing {bullet2.id} (score={score2}, sim={similarity:.3f}) "
                    f"in favor of {bullet1.id} (score={score1})"
                )
            else:
                # Keep bullet2, remove bullet1
                bullets_to_remove.add(bullet1.id)
                self.logger.debug(
                    f"Removing {bullet1.id} (score={score1}, sim={similarity:.3f}) "
                    f"in favor of {bullet2.id} (score={score2})"
                )

        # Remove bullets
        for bullet_id in bullets_to_remove:
            playbook.remove_bullet(bullet_id)

        removed_count = len(bullets_to_remove)
        self.logger.info(f"Removed {removed_count} duplicate bullets")

        return removed_count
