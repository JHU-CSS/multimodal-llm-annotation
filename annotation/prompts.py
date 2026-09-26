"""Prompts, label taxonomies, and templates used by the annotators."""

HASHTAG_OPTIONS = [
    "anime", "basketball", "booktok", "craft", "dance", "dogs",
    "food", "funnyvideos", "gaming", "gym", "homedecor", "makeup",
    "music", "skincare", "strangerthings", "tech", "travel", "wealth",
]

MELD_EMOTION_LABELS = ["joy", "sadness", "fear", "surprise", "disgust", "anger", "neutral"]
MELD_SENTIMENT_LABELS = ["positive", "negative", "neutral"]

VISION_PROMPT = """You are a TikTok short-form video classifier. Given a 3x3 frame grid and a transcript,
choose exactly one hashtag from the provided list (DO NOT invent new hashtags). Favor visuals over transcript if they conflict.
Respond ONLY with JSON: {"hashtag": "...", "confidence": 1-10, "rationale": "..."}"""

VISION_PROMPT_4X4 = """You are a TikTok short-form video classifier. Given a 4x4 frame grid and a transcript,
choose exactly one hashtag from the provided list (DO NOT invent new hashtags). Favor visuals over transcript if they conflict.
Respond ONLY with JSON: {"hashtag": "...", "confidence": 1-10, "rationale": "..."}"""

VISION_PROMPT_2X8 = """You are a TikTok short-form video classifier. Given a 2x8 frame grid (16 frames sampled in
chronological order from a short vertical video, laid out left-to-right across two rows) and a transcript,
choose exactly one hashtag from the provided list (DO NOT invent new hashtags). Favor visuals over transcript if they conflict.
Respond ONLY with JSON: {"hashtag": "...", "confidence": 1-10, "rationale": "..."}"""

VISION_PROMPT_IMAGE = """You are a TikTok short-form video classifier. Given a frame grid (frames sampled in
chronological order from a short video), choose exactly one hashtag from the provided list (DO NOT invent new hashtags).
Base your decision only on the visual content of the frames.
Respond ONLY with JSON: {"hashtag": "...", "confidence": 1-10, "rationale": "..."}"""

TRANSCRIPT_PROMPT = """You are a content classifier. Given a transcript, choose exactly one hashtag.
Respond ONLY with JSON: {"hashtag": "...", "confidence": 1-10, "rationale": "..."}"""

GEMINI_PROMPT_TEMPLATE = """You are a TikTok short-form video classifier.

Watch this video carefully and classify it into exactly ONE of the following 18 hashtag categories:
{options}

Instructions:
- Base your classification primarily on the visual content of the video.
- If there is speech/audio, use it as secondary evidence.
- Choose the single best-matching hashtag from the list above. Do NOT invent new hashtags.
- Provide a confidence score from 1 (very uncertain) to 10 (very confident).

Respond ONLY with valid JSON in this exact format:
{{"hashtag": "...", "confidence": <1-10>, "rationale": "one sentence explaining your choice"}}"""


MELD_VISION_PROMPT_3X3 = """You are a video analyst. You are given a 3x3 grid of frames from the TV show *Friends*, sampled in chronological order from a short video clip. Analyze the facial expressions in these frames and the tone of the transcript together to infer emotion.
The frames are arranged left to right, top to bottom, and separated by white lines.

# INPUT
You will receive:
- The name of one specific character (<CharacterName>) to analyze.
- A grid of 9 frames (visual input).

# CRITICAL INSTRUCTION
- You MUST use the exact character name provided in the prompt (<CharacterName>).
- Do NOT modify, abbreviate, replace, or omit it.
- Do NOT use any other character's name.
- Always include this exact name as the JSON key in your output.

# TASK
1. Determine whether the specified character (<CharacterName>) is **visible** in the 9-frame grid.
   - If clearly present in any of the frames (face or body visible), mark as visible.
   - If completely absent, mark as not visible.

2. If the character **is visible**, infer their **emotion** and **sentiment** using the image.

3. If the character **is not visible**, set both emotion and sentiment labels to "N/A" and confidences to null.

Allowed emotion labels: ["joy", "sadness", "fear", "surprise", "disgust", "anger", "neutral"]
Allowed sentiment labels: ["positive", "negative", "neutral"]

# OUTPUT FORMAT
Return **only valid JSON**, following this exact structure:

{
  "<CharacterName>": {
    "character_visible": <true or false>,
    "emotion": {
      "label": "<one of the 7 labels or 'N/A'>",
      "confidence": <integer 1-10 or null>
    },
    "sentiment": {
      "label": "<one of the 3 labels or 'N/A'>",
      "confidence": <integer 1-10 or null>
    }
  }
}
"""


MELD_VISION_PROMPT_4X4_IMAGE = """You are a video analyst. You are given a 4x4 grid of frames from the TV show *Friends*, sampled in chronological order from a short video clip. Analyze the facial expressions in these frames to infer emotion.
The frames are arranged left to right, top to bottom, and separated by white lines.

# INPUT
You will receive:
- The name of one specific character (<CharacterName>) to analyze.
- A grid of 16 frames (visual input).

# CRITICAL INSTRUCTION
- You MUST use the exact character name provided in the prompt (<CharacterName>).
- Do NOT modify, abbreviate, replace, or omit it.
- Do NOT use any other character's name.
- Always include this exact name as the JSON key in your output.

# TASK
1. Determine whether the specified character (<CharacterName>) is **visible** in the 16-frame grid.
   - If clearly present in any of the frames (face or body visible), mark as visible.
   - If completely absent, mark as not visible.

2. If the character **is visible**, infer their **emotion** and **sentiment** using the image.

3. If the character **is not visible**, set both emotion and sentiment labels to "N/A" and confidences to null.

Allowed emotion labels: ["joy", "sadness", "fear", "surprise", "disgust", "anger", "neutral"]
Allowed sentiment labels: ["positive", "negative", "neutral"]

# OUTPUT FORMAT
Return **only valid JSON**, following this exact structure:

{
  "<CharacterName>": {
    "character_visible": <true or false>,
    "emotion": {
      "label": "<one of the 7 labels or 'N/A'>",
      "confidence": <integer 1-10 or null>
    },
    "sentiment": {
      "label": "<one of the 3 labels or 'N/A'>",
      "confidence": <integer 1-10 or null>
    }
  }
}
"""


MELD_TEXT_PROMPT = """You are a dialogue analyst. You are given the name of one character from the TV show *Friends* and their single line of dialogue from a short clip. Infer that character's emotion and sentiment from the text alone.

# INPUT
You will receive:
- The name of one specific character (<CharacterName>) to analyze.
- A short text snippet (the corresponding dialogue).

# CRITICAL INSTRUCTION
- You MUST use the exact character name provided in the prompt (<CharacterName>).
- Do NOT modify, abbreviate, replace, or omit it.
- Always include this exact name as the JSON key in your output.

# TASK
Infer the specified character's **emotion** and **sentiment** from the dialogue text.

Allowed emotion labels: ["joy", "sadness", "fear", "surprise", "disgust", "anger", "neutral"]
Allowed sentiment labels: ["positive", "negative", "neutral"]

# OUTPUT FORMAT
Return **only valid JSON**, following this exact structure:

{
  "<CharacterName>": {
    "emotion": {
      "label": "<one of the 7 labels>",
      "confidence": <integer 1-10>
    },
    "sentiment": {
      "label": "<one of the 3 labels>",
      "confidence": <integer 1-10>
    }
  }
}
"""


MELD_VISION_PROMPT_3X3_TEXT = """You are a video analyst. You are given a 3x3 grid of frames from the TV show *Friends*, sampled in chronological order from a short video clip, together with the corresponding dialogue text. Analyze the facial expressions in these frames and the tone of the transcript together to infer emotion.
The frames are arranged left to right, top to bottom, and separated by white lines.

# INPUT
You will receive:
- The name of one specific character (<CharacterName>) to analyze.
- A short text snippet (the corresponding dialogue).
- A grid of 9 frames (visual input).

# CRITICAL INSTRUCTION
- You MUST use the exact character name provided in the prompt (<CharacterName>).
- Do NOT modify, abbreviate, replace, or omit it.
- Do NOT use any other character's name.
- Always include this exact name as the JSON key in your output.

# TASK
1. Determine whether the specified character (<CharacterName>) is **visible** in the 9-frame grid.
   - If clearly present in any of the frames (face or body visible), mark as visible.
   - If completely absent, mark as not visible.

2. If the character **is visible**, infer their **emotion** and **sentiment** using both the image and the text.

3. If the character **is not visible**, set both emotion and sentiment labels to "N/A" and confidences to null.

Allowed emotion labels: ["joy", "sadness", "fear", "surprise", "disgust", "anger", "neutral"]
Allowed sentiment labels: ["positive", "negative", "neutral"]

# OUTPUT FORMAT
Return **only valid JSON**, following this exact structure:

{
  "<CharacterName>": {
    "character_visible": <true or false>,
    "emotion": {
      "label": "<one of the 7 labels or 'N/A'>",
      "confidence": <integer 1-10 or null>
    },
    "sentiment": {
      "label": "<one of the 3 labels or 'N/A'>",
      "confidence": <integer 1-10 or null>
    }
  }
}
"""


MELD_VISION_PROMPT_4X4 = """You are a video analyst. You are given a 4x4 grid of frames from the TV show *Friends*, sampled in chronological order from a short video clip. Analyze the facial expressions in these frames and the tone of the transcript together to infer emotion.
The frames are arranged left to right, top to bottom, and separated by white lines.

# INPUT
You will receive:
- The name of one specific character (<CharacterName>) to analyze.
- A short text snippet (the corresponding dialogue).
- A grid of 16 frames (visual input).

# CRITICAL INSTRUCTION
- You MUST use the exact character name provided in the prompt (<CharacterName>).
- Do NOT modify, abbreviate, replace, or omit it.
- Do NOT use any other character's name.
- Always include this exact name as the JSON key in your output.

# TASK
1. Locate the specified character (<CharacterName>) in the frames.

2. Infer that specific character's **emotion** and **sentiment** using both image and text.

Allowed emotion labels: ["joy", "sadness", "fear", "surprise", "disgust", "anger", "neutral"]
Allowed sentiment labels: ["positive", "negative", "neutral"]

# OUTPUT FORMAT
Return **only valid JSON**, following this exact structure:

{
  "<CharacterName>": {
    "emotion": {
      "label": "<one of the 7 labels>",
      "confidence": <integer 1-10>
    },
    "sentiment": {
      "label": "<one of the 3 labels>",
      "confidence": <integer 1-10>
    }
  }
}
"""
