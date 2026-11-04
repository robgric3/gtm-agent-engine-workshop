import json
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as module


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cache = patch.dict(module.data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.score = module.ProspectScore(
            score=80,
            justification="Good fit with some required technologies missing.",
            rubric_breakdown=module.RubricBreakdown(
                revenue_fit=90, tech_stack_match=70, segment_fit=80
            ),
        )
        self.scorer = Mock()
        self.scorer.invoke.return_value = self.score
        self.scoring_patch = patch.object(module, "_scoring_llm", self.scorer)
        self.scoring_patch.start()
        self.addCleanup(self.scoring_patch.stop)
        self.ids = {"prospect_id": "LEAD-70001", "offering_id": "OFFER-10001"}

    def scoring_records(self):
        content = self.scorer.invoke.call_args.args[0][1]["content"]
        offering, profile = content.split("\n\nProspect profile:\n")
        return json.loads(offering.removeprefix("Offering:\n")), json.loads(profile)

    def test_score_loads_records_by_id(self):
        self.assertEqual(
            module.score_prospect.invoke(self.ids), self.score.model_dump()
        )
        offering, profile = self.scoring_records()
        self.assertEqual(
            offering, module.data_service.get_offering(self.ids["offering_id"])
        )
        self.assertEqual(profile["prospect_id"], self.ids["prospect_id"])
        self.assertEqual(
            profile,
            module.build_prospect_profile.invoke(
                {"prospect_id": self.ids["prospect_id"]}
            )["prospect_profile"],
        )

    def test_missing_prospect(self):
        self.assertEqual(
            module.score_prospect.invoke({**self.ids, "prospect_id": "missing"}),
            {"score": None, "error": "Prospect not found."},
        )
        self.scorer.invoke.assert_not_called()
        self.assertEqual(
            module.build_prospect_profile.invoke({"prospect_id": "missing"}),
            {"prospect_profile": None, "found": False},
        )

    def test_invalid_offering(self):
        offering = module.data_service.get_offering(self.ids["offering_id"])
        invalid = [None] + [
            {key: value for key, value in offering.items() if key != field}
            for field in ("required_tech_stack", "min_annual_revenue", "description")
        ]
        for record in invalid:
            with (
                self.subTest(offering=record),
                patch.object(module.data_service, "get_offering", return_value=record),
            ):
                self.assertEqual(
                    module.score_prospect.invoke(self.ids),
                    {"score": None, "error": "Cannot score without a valid offering."},
                )
        self.scorer.invoke.assert_not_called()

    def test_cached_profile_keeps_current_tech_stack_for_scoring(self):
        profile = module.build_prospect_profile.invoke(
            {"prospect_id": self.ids["prospect_id"]}
        )["prospect_profile"]
        with patch.object(
            module.data_service, "fetch_tech_stack", return_value=["new technology"]
        ):
            module.score_prospect.invoke(self.ids)
        _, scored_profile = self.scoring_records()
        self.assertEqual(scored_profile, {**profile, "tech_stack": ["new technology"]})
        self.assertEqual(
            module.build_prospect_profile.invoke(
                {"prospect_id": self.ids["prospect_id"]}
            )["prospect_profile"],
            profile,
        )

    def run_graph(self, responses, request):
        model = ToolCallingModel(responses=responses)
        graph = module.create_deep_agent(
            model=model,
            tools=[
                module.score_prospect,
                module.build_prospect_profile,
                module.lookup_offering,
                module.get_current_rep,
                module.get_prospect,
                module.send_prospect_email,
            ],
            system_prompt=module.SYSTEM_PROMPT,
        )
        result = graph.invoke(
            {"messages": [{"role": "user", "content": request}]},
            config={"metadata": {"user_id": "rep_amills"}},
        )
        return result["messages"]

    def test_direct_score_completes_in_two_model_turns(self):
        messages = self.run_graph(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "score_prospect", "args": self.ids, "id": "score"}
                    ],
                ),
                AIMessage(content="The prospect scores 80/100."),
            ],
            "Score LEAD-70001 against OFFER-10001.",
        )
        self.assertEqual(sum(isinstance(m, AIMessage) for m in messages), 2)
        results = [m for m in messages if isinstance(m, ToolMessage)]
        self.assertEqual([m.name for m in results], ["score_prospect"])
        self.assertEqual(json.loads(results[0].content), self.score.model_dump())

    def test_email_resolves_sender_without_rep_lookup(self):
        messages = self.run_graph(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_prospect",
                            "args": {"prospect_id": "LEAD-70001"},
                            "id": "contact",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "send_prospect_email",
                            "args": {
                                "prospect": {
                                    "name": "Maria Garcia",
                                    "email": "maria@example.com",
                                },
                                "subject": "Follow up",
                                "body": "Let's connect.",
                            },
                            "id": "email",
                        }
                    ],
                ),
                AIMessage(content="Email sent."),
            ],
            "Send a follow-up email to LEAD-70001.",
        )
        results = [m for m in messages if isinstance(m, ToolMessage)]
        self.assertEqual(
            [m.name for m in results], ["get_prospect", "send_prospect_email"]
        )
        email = json.loads(results[-1].content)
        self.assertEqual(email["status"], "sent")
        self.assertEqual(email["from"], "ava.mills@northpoint.com")
        self.assertEqual(email["from_name"], "Ava Mills")


if __name__ == "__main__":
    unittest.main()
