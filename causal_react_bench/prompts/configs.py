EVENT_RULES = """
Each output prompt must contain 2 to 4 distinct, temporally ordered events.

An event is a visually observable action or state transition of a character,
object, animal, or other scene entity.

Camera motion does NOT count toward the required 2-4 events.

Camera handling:
- simple camera motion such as a static shot, slow pan, tilt, tracking motion,
  or gentle zoom may be preserved;
- complicated, distracting, or multi-stage camera choreography may be simplified
  into a natural short camera description;
- do not reject an otherwise useful prompt merely because camera motion exists;
- do not invent new scene actions while simplifying camera descriptions.

The output does not need to explicitly list or annotate the events.
They only need to be clearly expressed in the natural-language prompt.
"""

SYS_SHORT = f"""
You curate source captions for short event-driven video generation.

The source duration is at most 5 seconds.

{EVENT_RULES}

Create one usable prompt only if a coherent sequence of 2-4 visible events can
naturally fit within the supplied source duration.

Requirements:
- preserve the observable source semantics;
- preserve important subjects, objects, and scene context;
- keep a coherent short scene;
- if the source contains more than 4 events, retain a coherent 2-4 event subset;
- do not invent missing actions;
- reject if fewer than 2 supported events remain;
- reject physically implausible timing;
- remove or simplify redundant narrative, dialogue, or interpretation.

Minimally rewrite when useful.

Return JSON only:
{{
  "valid": true,
  "prompts": [{{"prompt": "..."}}],
  "reason": ""
}}
"""

SYS_TRIM = f"""
You convert a 5-7 second source caption into one physically plausible
5-second video-generation prompt.

{EVENT_RULES}

Select the strongest coherent sequence of 2-4 visible events and rewrite it
into a natural 5-second scene.

Requirements:
- preserve source semantics;
- preserve important subjects and objects;
- remove peripheral events instead of unnaturally accelerating them;
- keep a coherent scene;
- do not invent actions, objects, locations, or causal relations;
- simplify complicated camera descriptions when needed;
- the result must naturally fit within 5 seconds.

Return JSON only:
{{
  "valid": true,
  "prompts": [{{"prompt": "..."}}],
  "reason": ""
}}
"""

SYS_SPLIT = f"""
You decompose a source caption longer than 7 seconds into independent prompts
suitable for 5-second video generation.

{EVENT_RULES}

Each output prompt must:
- contain a coherent sequence of 2-4 supported visible events;
- be independently understandable;
- be naturally realizable within about 5 seconds;
- preserve source subjects, objects, actions, and scene semantics;
- avoid combining content separated by a major time jump or unrelated scene;
- simplify complicated camera motion when necessary;
- not invent new actions.

Extract multiple useful short semantic segments when supported by the source.
Do not force unusable source material into an output segment.

Return JSON only:
{{
  "valid": true,
  "prompts": [
    {{"prompt": "..."}}
  ],
  "reason": ""
}}
"""

SYS_REVIEW = f"""
You independently review a derived event-driven video prompt for TARGET_SECONDS.

{EVENT_RULES}

Compare DERIVED_PROMPT against SOURCE_PROMPT.

The derived prompt is valid only if:
- its visible scene content is supported by the source;
- it contains 2-4 distinct, temporally ordered entity events;
- the sequence is coherent and physically plausible within TARGET_SECONDS;
- important subjects and objects remain consistent;
- no unsupported action, object, location, or causal relation was invented;
- camera motion, if present, is reasonable for the short scene.

Simple camera motion is acceptable.
Complex camera motion may be minimally simplified.

If the prompt is already good, return it unchanged.
If a small wording or camera-description repair is sufficient, minimally repair it.
Do not add a new event to make an invalid sample pass.
Otherwise reject it.

Return JSON only:
{{
  "valid": true,
  "faithful": true,
  "single_scene": true,
  "physically_plausible": true,
  "fits_duration": true,
  "prompt": "...",
  "reason": ""
}}
"""
