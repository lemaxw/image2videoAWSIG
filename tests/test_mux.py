import unittest
from pathlib import Path
from unittest.mock import patch

from services.orchestrator.mux import mux_video_audio


class MuxTests(unittest.TestCase):
    @patch("services.orchestrator.mux.subprocess.run")
    def test_final_encode_has_unambiguous_social_media_first_frame(self, run):
        mux_video_audio(
            Path("generated.mp4"),
            Path("audio.wav"),
            Path("final.mp4"),
        )

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-bf") + 1], "0")
        self.assertEqual(command[command.index("-g") + 1], "30")
        self.assertEqual(command[command.index("-keyint_min") + 1], "30")
        self.assertEqual(command[command.index("-sc_threshold") + 1], "0")
        self.assertEqual(command[command.index("-profile:v") + 1], "main")
        self.assertEqual(command[command.index("-maxrate") + 1], "6M")
        self.assertEqual(command[command.index("-bufsize") + 1], "12M")
        self.assertEqual(command[command.index("-fps_mode") + 1], "cfr")
        self.assertEqual(command[command.index("-video_track_timescale") + 1], "30000")
        self.assertEqual(command[command.index("-ar") + 1], "44100")
        self.assertEqual(command[command.index("-ac") + 1], "2")
        self.assertEqual(command[command.index("-use_editlist") + 1], "0")
        self.assertEqual(command[command.index("-movflags") + 1], "+faststart")
        self.assertEqual(command[command.index("-brand") + 1], "mp42")
        self.assertEqual(command[command.index("-t") + 1], "5.000")
        filter_complex = command[command.index("-filter_complex") + 1]
        self.assertIn("aresample=44100", filter_complex)
        self.assertIn("asetpts=PTS-STARTPTS+1024/SR/TB", filter_complex)


if __name__ == "__main__":
    unittest.main()
