import unittest

import app


class WebRtcConfigurationTests(unittest.TestCase):
    def test_uses_public_stun_without_turn_secrets(self):
        configuration, turn_configured = app.build_rtc_configuration({})

        self.assertFalse(turn_configured)
        self.assertEqual(
            configuration,
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )

    def test_adds_turn_server_only_when_credentials_are_complete(self):
        secrets = {
            "WEBRTC_TURN_URLS": [
                "turn:turn.example.test:3478?transport=tcp ",
                "",
            ],
            "WEBRTC_TURN_USERNAME": "user",
            "WEBRTC_TURN_CREDENTIAL": "secret",
        }

        configuration, turn_configured = app.build_rtc_configuration(secrets)

        self.assertTrue(turn_configured)
        self.assertEqual(
            configuration["iceServers"][1],
            {
                "urls": ["turn:turn.example.test:3478?transport=tcp"],
                "username": "user",
                "credential": "secret",
            },
        )

    def test_does_not_add_incomplete_turn_credentials(self):
        configuration, turn_configured = app.build_rtc_configuration({
            "WEBRTC_TURN_URLS": ["turn:turn.example.test:3478"],
            "WEBRTC_TURN_USERNAME": "user",
        })

        self.assertFalse(turn_configured)
        self.assertEqual(len(configuration["iceServers"]), 1)


if __name__ == "__main__":
    unittest.main()
