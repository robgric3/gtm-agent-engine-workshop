import copy
import json
import unittest
from types import MappingProxyType
from unittest.mock import patch

from gtm_agent import data_service

with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.records = patch.object(data_service, "PROSPECTS", copy.deepcopy(data_service.PROSPECTS))
        self.records.start()
        self.addCleanup(self.records.stop)
        self.cache = patch.object(data_service, "_PROFILES", {})
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.tracing = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def test_update_persists_technology(self):
        result = data_service.update_prospect_info("LEAD-39002", "Terraform")

        self.assertTrue(result["updated"])
        self.assertTrue(result["found"])
        self.assertIn("Terraform", result["tech_stack"])
        self.assertEqual(result["tech_stack"], data_service.fetch_tech_stack("LEAD-39002"))

    def test_update_rebuilds_cached_profile(self):
        before = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})
        self.assertNotIn("Terraform", before["prospect_profile"]["tech_stack"])

        gtm_agent.update_prospect_info.invoke({"prospect_id": "LEAD-39002", "technology": "Terraform"})
        after = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})

        self.assertTrue(after["found"])
        self.assertIn("Terraform", after["prospect_profile"]["tech_stack"])
        self.assertEqual(after, gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"}))

    def test_scoring_reads_persisted_stack_even_with_old_profile(self):
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})["prospect_profile"]
        offering = data_service.get_offering("OFFER-10004")
        data_service.update_prospect_info("LEAD-39002", "Terraform")

        def score(messages):
            saved_profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
            self.assertIn("Terraform", saved_profile["tech_stack"])
            missing = set(offering["required_tech_stack"]) - set(saved_profile["tech_stack"])
            self.assertNotIn("Terraform", missing)
            return gtm_agent.ProspectScore(
                score=90,
                justification=f"Missing technologies: {sorted(missing)}",
                rubric_breakdown=gtm_agent.RubricBreakdown(
                    revenue_fit=90, tech_stack_match=100, segment_fit=90,
                ),
            )

        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score
            result = gtm_agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 100)

    def test_repeated_update_does_not_duplicate_technology(self):
        data_service.update_prospect_info("LEAD-39002", "Terraform")
        data_service.update_prospect_info("LEAD-39002", "Terraform")

        self.assertEqual(data_service.fetch_tech_stack("LEAD-39002").count("Terraform"), 1)

    def test_unknown_prospect_is_not_found(self):
        self.assertEqual(
            data_service.update_prospect_info("unknown", "Terraform"),
            {"updated": False, "found": False},
        )
        self.assertIsNone(data_service.get_prospect_record("unknown"))

    def test_failed_write_raises_without_changing_saved_stack(self):
        record = data_service.get_prospect_record("LEAD-39002")
        with patch.dict(data_service.PROSPECTS, {"LEAD-39002": MappingProxyType(record)}):
            with self.assertRaises(TypeError):
                data_service.update_prospect_info("LEAD-39002", "Terraform")

            self.assertNotIn("Terraform", data_service.fetch_tech_stack("LEAD-39002"))


if __name__ == "__main__":
    unittest.main()
