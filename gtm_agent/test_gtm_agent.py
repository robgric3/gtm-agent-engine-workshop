import importlib
import json
import os
import unittest
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "LANGSMITH_TRACING": "false",
        })
        cls.environment.start()
        cls.addClassCleanup(cls.environment.stop)
        with patch("dotenv.load_dotenv"):
            cls.module = importlib.import_module("gtm_agent.gtm_agent")

    def test_agent_uses_only_gtm_prompt_and_tools(self):
        requests = []

        def respond(messages, **kwargs):
            requests.append((messages, kwargs))
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Ready"))])

        with patch.object(ChatOpenAI, "_generate", side_effect=respond):
            result = self.module.run_agent("Hello", user_id="rep_amills")

        self.assertEqual(result["reply"], "Ready")
        messages, kwargs = requests[0]
        self.assertEqual(
            [message.content for message in messages if isinstance(message, SystemMessage)],
            [self.module.SYSTEM_PROMPT],
        )
        self.assertEqual({tool["function"]["name"] for tool in kwargs["tools"]}, {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info",
            "get_current_rep",
        })

    def test_runtime_receives_metadata_and_email_uses_signed_in_rep(self):
        responses = iter([
            AIMessage(content="", tool_calls=[{
                "name": "get_current_rep", "args": {}, "id": "rep",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email", "id": "email", "args": {
                    "prospect": {"name": "Test Prospect", "email": "prospect@example.com"},
                    "subject": "Hello", "body": "Following up",
                },
            }]),
            AIMessage(content="Sent"),
        ])
        configs = []
        original = self.module.get_current_rep.func

        def get_rep(runtime):
            configs.append(runtime.config)
            return original(runtime)

        def respond(*args, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=next(responses))])

        with patch.object(ChatOpenAI, "_generate", side_effect=respond), patch.object(
            self.module.get_current_rep, "func", side_effect=get_rep,
        ):
            result = self.module.run_agent(
                "Email the prospect", user_id="rep_dweiss",
                environment="test", thread_id="test-thread",
            )

        expected_metadata = {
            "thread_id": "test-thread", "user_id": "rep_dweiss",
            "environment": "test", "request_intent": "send_email",
        }
        self.assertEqual(
            {key: configs[0]["metadata"][key] for key in expected_metadata},
            expected_metadata,
        )
        outputs = [json.loads(message.content) for message in result["messages"]
                   if isinstance(message, ToolMessage)]
        self.assertEqual(outputs[0]["rep"], self.module.data_service.get_rep("rep_dweiss"))
        self.assertEqual(outputs[1]["status"], "sent")
        self.assertEqual(outputs[1]["from"], outputs[0]["rep"]["email"])
        self.assertEqual(outputs[1]["to"], "prospect@example.com")

    def test_run_agent_preserves_invoke_config(self):
        run_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
        with patch.object(self.module.uuid, "uuid4", return_value=run_id), patch.object(
            self.module.gtm_agent, "invoke",
            return_value={"messages": [AIMessage(content="Done")]},
        ) as invoke:
            result = self.module.run_agent(
                "Look up offering", user_id="rep_amills",
                environment="test", thread_id="thread",
            )
        self.assertEqual(invoke.call_args.kwargs["config"], {
            "run_name": "GTM Assistant", "run_id": run_id,
            "metadata": {
                "thread_id": "thread", "user_id": "rep_amills",
                "environment": "test", "request_intent": "lookup_offering",
            },
        })
        self.assertEqual(result["run_id"], str(run_id))

    def test_documented_install_does_not_require_deepagents(self):
        install = self.module.__doc__.split("Install:", 1)[1]
        self.assertNotIn("deepagents", install.split())


if __name__ == "__main__":
    unittest.main()
