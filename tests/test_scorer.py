from doubt import scorer


def test_rederivation_sees_the_tool_output_that_shaped_the_next_step():
    messages = [
        {"role": "user", "content": "encode in.mov to HEVC"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "terminal", "arguments": '{"command": "ffmpeg -preset slowest"}'}}]},
        {"role": "tool", "content": "Unable to parse option value \"slowest\""},
    ]
    text = scorer.context_transcript(messages)
    assert "TOOL OUTPUT: Unable to parse option value" in text
    assert text.index("[called terminal") < text.index("TOOL OUTPUT")
