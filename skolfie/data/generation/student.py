"""
skolfie/data/generation/student.py
-----------------------------------
Student-essay prompts for generating modern AI text that matches what
teachers see (added 2026-10-29, after essays from current Claude models and
Argon scored 0% AI: the corpus only had GPT-3.5/GPT-4-era AI text).

Two halves:

- `PERSUADE_PROMPTS`: the 15 assignments of the PERSUADE 2.0 corpus
  (25,000 essays by US students in grades 6-12, CC BY 4.0), so the AI
  essays share their topics with the human student essays and the model
  cannot learn "this topic = AI".
- the existing bank's academic essays and creative personal narratives
  (`skolfie.data.generation.prompts`), for college-level writing.

Each prompt is combined with a student persona (grade 6-12 or college), an
optional "make it sound human" instruction (what students actually ask for)
and a word target. `student_prompt_plan(n, seed)` is deterministic, and its
specs are ordinary `PromptSpec`s, so `scripts/generate_ai_texts.py`'s
`make_record` turns them into fresh-source records.
"""

from __future__ import annotations

import random

from skolfie.data.generation.prompts import (
    OUTPUT_INSTRUCTION,
    PromptSpec,
    max_tokens_for,
    sample_prompts,
)

# PERSUADE 2.0 prompt names -> the assignment as a student would get it.
PERSUADE_PROMPTS: dict[str, str] = {
    "phones-and-driving": (
        "Write an essay arguing whether drivers should or should not be able to use cell "
        "phones in any capacity while operating a vehicle."),
    "car-free-cities": (
        "Write an explanatory essay to inform fellow citizens about the advantages of limiting "
        "car usage, using ideas from articles about car-free neighborhoods in Germany, driving "
        "bans in Paris and car-free days in Bogota."),
    "summer-projects": (
        "Some schools require students to complete summer projects to keep learning during "
        "the break. Should these projects be teacher-designed or student-designed? Write an "
        "essay taking a position."),
    "cowboy-who-rode-the-waves": (
        "You have read the article 'A Cowboy Who Rode the Waves' about Luke Bomberger, who "
        "joined the Seagoing Cowboys program after World War II. Write an essay from Luke's "
        "point of view convincing others to participate in the program."),
    "mandatory-extracurricular-activities": (
        "Your principal has decided that all students must participate in at least one "
        "extracurricular activity. Do you agree or disagree with this decision? Write an essay "
        "explaining your position."),
    "exploring-venus": (
        "In 'The Challenge of Exploring Venus' the author suggests that studying Venus is a "
        "worthy pursuit despite the dangers it presents. Write an essay evaluating how well the "
        "author supports this idea."),
    "facial-action-coding-system": (
        "In the article 'Making Mona Lisa Smile', the author describes how a new technology "
        "called the Facial Action Coding System lets computers identify human emotions. Write "
        "an essay arguing whether using this technology to read students' emotional "
        "expressions in a classroom is valuable."),
    "face-on-mars": (
        "You have read the article 'Unmasking the Face on Mars'. Imagine you are a NASA "
        "scientist talking with someone who thinks the Face was created by aliens. Write an "
        "argumentative essay convincing them that the Face is just a natural landform."),
    "community-service": (
        "Some people think that students should be required to perform community service. "
        "Others think it should be voluntary. Write a letter to your principal taking a "
        "position on whether students should be required to perform community service."),
    "grades-for-extracurricular-activities": (
        "Your principal is considering a policy that students must have at least a grade B "
        "average to participate in sports or other activities. Write a letter to your "
        "principal arguing for or against this policy."),
    "driverless-cars": (
        "In the article 'Driverless Cars Are Coming', the author presents both positive and "
        "negative aspects of driverless cars. Write an essay arguing for or against the "
        "development of these cars."),
    "electoral-college": (
        "Write a letter to your state senator arguing in favor of keeping the Electoral College "
        "or changing to election by popular vote for the president of the United States."),
    "cell-phones-at-school": (
        "Your principal is reconsidering the school's cell phone policy: allowing phones during "
        "lunch and free periods, or not allowing phones at school at all. Write a letter to "
        "your principal stating which policy you believe is better and why."),
    "distance-learning": (
        "Some schools offer distance learning as an option for students to attend classes from "
        "home online or by video conferencing. Would students benefit from attending classes "
        "from home? Write an essay taking a position."),
    "seeking-multiple-opinions": (
        "When people ask for advice, they sometimes talk to more than one person. Write an "
        "essay explaining why seeking multiple opinions can help someone make a better choice."),
}

# Personas: (id, instruction). Grades match PERSUADE (6-12); college for the bank prompts.
SCHOOL_PERSONAS = tuple(
    (f"grade{g}", f"Write it as {'an' if g in (8, 11) else 'a'} {g}th-grade student in the "
                  "United States would.")
    for g in range(6, 13)
)
COLLEGE_PERSONAS = (
    ("college1", "Write it as a first-year college student would."),
    ("college-esl", "Write it as a college student who is a fluent but non-native speaker of "
                    "English would."),
)
# What students add to their requests; "" means no extra instruction.
HUMANIZE = (
    ("plain", ""),
    ("plain", ""),
    ("human-natural", "Make it sound like a real student wrote it, not an AI: no generic "
                      "opening line, no tidy summary at the end."),
    ("human-casual", "Make it sound natural and not like AI: vary the sentence length, use a "
                     "few contractions, and don't make it too polished."),
)
PERSUADE_LENGTHS = (250, 350, 450, 600)
TEMPERATURES = (0.7, 1.0)


def _persuade_spec(rng: random.Random, idx: int) -> PromptSpec:
    name = rng.choice(sorted(PERSUADE_PROMPTS))
    persona_id, persona = rng.choice(SCHOOL_PERSONAS)
    style_id, style = rng.choice(HUMANIZE)
    words = rng.choice(PERSUADE_LENGTHS)
    temperature = rng.choice(TEMPERATURES)
    parts = [PERSUADE_PROMPTS[name], persona]
    if style:
        parts.append(style)
    parts += [f"Aim for about {words} words.", OUTPUT_INSTRUCTION]
    prompt_id = (f"persuade-{name}.{persona_id}.{style_id}.w{words}."
                 f"t{round(temperature * 10):02d}.n{idx:04d}")
    return PromptSpec(prompt_id=prompt_id, template_id=f"persuade-{name}", domain="academic",
                      subdomain="student_essay", style=f"{persona_id}+{style_id}",
                      text=" ".join(parts), target_words=words, temperature=temperature,
                      max_tokens=max_tokens_for(words))


def student_prompt_plan(n: int, seed: int = 0, persuade_share: float = 0.6) -> list[PromptSpec]:
    """`n` prompts: about `persuade_share` PERSUADE assignments, the rest college essays and
    personal narratives from the bank (with a college persona added). Deterministic."""
    if n < 0 or not 0 <= persuade_share <= 1:
        raise ValueError("n must be >= 0 and persuade_share in [0, 1]")
    rng = random.Random(seed)
    n_persuade = round(n * persuade_share)
    specs = [_persuade_spec(rng, i) for i in range(n_persuade)]
    bank = [s for s in sample_prompts(4 * (n - n_persuade) + 10, seed=seed,
                                      domains=["academic", "creative"])
            if s.subdomain in ("essay", "explanation", "personal_narrative")]
    for spec in bank[: n - n_persuade]:
        persona_id, persona = rng.choice(COLLEGE_PERSONAS)
        text = spec.text.replace(OUTPUT_INSTRUCTION, f"{persona} {OUTPUT_INSTRUCTION}")
        specs.append(PromptSpec(prompt_id=f"{spec.prompt_id}.{persona_id}",
                                template_id=spec.template_id, domain=spec.domain,
                                subdomain=spec.subdomain, style=f"{spec.style}+{persona_id}",
                                text=text, target_words=spec.target_words,
                                temperature=spec.temperature, max_tokens=spec.max_tokens,
                                sample_idx=spec.sample_idx))
    rng.shuffle(specs)
    return specs
