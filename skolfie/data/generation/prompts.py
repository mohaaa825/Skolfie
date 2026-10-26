"""
skolfie/data/generation/prompts.py
-----------------------------------
The prompt bank used to generate fresh AI text, plus deterministic sampling.

Prompts are plain data: each entry of `PROMPT_TEMPLATES` is a dict with an
`id`, a schema `domain` (one of `skolfie.data.schema.DOMAINS`), a
`subdomain` (the prompt type, e.g. "essay" or "forum_post"), a `template`
with `{slot}` placeholders, and `slots`, mapping each placeholder to a list
in `SLOT_VALUES`. A template may also carry `lengths`, the word targets that
suit it (a tweet-length post should not be asked for 600 words).

One concrete prompt is a template with its slots filled, combined with a
prompting style from `STYLES` (plain instruction, a persona, or "write like
a human, not an AI"), a word target and a sampling temperature. Its
`prompt_id` encodes all of those choices, so the same id always means the
same prompt text, and `sample_prompts(n, seed)` always returns the same plan.

The four target domains (academic, creative, technical, casual) get equal
weight; review and news get half weight each.
"""

from __future__ import annotations

import random
import string
from collections import Counter
from dataclasses import dataclass
from math import prod
from typing import Iterable

from skolfie.data.schema import DOMAINS

TARGET_DOMAINS = ("academic", "creative", "technical", "casual")

# Per balanced block of 10 sampled prompts: 2 of each target domain, 1 review, 1 news.
DOMAIN_BLOCK = ("academic", "creative", "technical", "casual") * 2 + ("review", "news")

DEFAULT_LENGTHS = (100, 200, 300, 450, 600)
TEMPERATURES = (0.6, 0.8, 1.0)

# Appended to every prompt: asks for the text only, so less output needs
# post-filtering. Models still add preambles sometimes; the CLI strips those.
OUTPUT_INSTRUCTION = "Reply with the text itself only: no title, no preface and no notes after it."

# --------------------------------------------------------------------------- #
# Slot values
# --------------------------------------------------------------------------- #

SLOT_VALUES: dict[str, list[str]] = {
    # academic
    "essay_questions": [
        "should university tuition be free for all students",
        "does social media do more harm than good to teenagers",
        "should voting be compulsory in democracies",
        "is nuclear power necessary to reach net-zero emissions",
        "should homework be abolished in primary schools",
        "do standardized tests measure anything meaningful",
        "should cities ban private cars from their centres",
        "is remote work better for society than office work",
        "should zoos exist in the twenty-first century",
        "does economic growth require rising inequality",
        "should governments regulate artificial intelligence the way they regulate medicine",
        "is it ethical to eat meat",
        "should the voting age be lowered to sixteen",
        "do museums have a duty to return artefacts taken during colonial rule",
        "is space exploration worth its cost",
        "should a four-day working week become the norm",
    ],
    "academic_topics": [
        "the causes of the French Revolution",
        "the role of the printing press in the Reformation",
        "how the Industrial Revolution changed family life",
        "Keynesian and monetarist responses to recessions",
        "the bystander effect and its critics",
        "the origins of the Cold War",
        "the ethics of utilitarianism versus deontology",
        "the impact of the Green Revolution on food security",
        "the representation of women in Victorian novels",
        "urbanisation in nineteenth-century Britain",
        "the causes and consequences of the 2008 financial crisis",
        "the influence of Roman law on modern legal systems",
        "the decline of the Mayan civilisation",
        "nature versus nurture in language acquisition",
        "the Harlem Renaissance and American identity",
    ],
    "courses": [
        "first-year history",
        "introductory sociology",
        "political philosophy",
        "environmental science",
        "developmental psychology",
        "comparative literature",
        "economics 101",
        "public health",
    ],
    "research_topics": [
        "a sociology study of how remote work affects junior employees' skill development",
        "an ecology study of urban green roofs as habitat for pollinators",
        "a psychology experiment on sleep deprivation and moral decision-making",
        "an education study comparing handwritten and typed lecture notes",
        "a public-health survey of e-scooter injuries in three European cities",
        "a computer-science paper on detecting duplicated code in student assignments",
        "an economics paper on minimum-wage increases and small restaurant closures",
        "a linguistics study of code-switching among bilingual teenagers on social media",
        "a materials-science paper on recycled plastic as a concrete additive",
        "a medical study of music therapy for patients recovering from stroke",
        "a political-science analysis of local newspaper closures and voter turnout",
        "a marine-biology survey of microplastics in coastal shellfish",
        "a nutrition study of breakfast skipping and concentration in secondary school pupils",
        "a history paper on letters written by soldiers' families in the First World War",
    ],
    "academic_concepts": [
        "opportunity cost",
        "natural selection",
        "the placebo effect",
        "confirmation bias",
        "supply and demand",
        "the greenhouse effect",
        "photosynthesis",
        "the separation of powers",
        "statistical significance",
        "cognitive dissonance",
        "plate tectonics",
        "the tragedy of the commons",
        "entropy",
        "comparative advantage",
        "the social contract",
    ],
    "academic_audiences": [
        "a classmate who missed the lecture",
        "first-year undergraduates",
        "a curious general reader",
        "students revising for an exam",
    ],
    # creative
    "story_premises": [
        "a lighthouse keeper who receives a letter addressed to someone who died fifty years ago",
        "two strangers stuck in a lift during a city-wide blackout",
        "a girl who finds out her grandmother was once a famous jazz singer",
        "a delivery driver who keeps getting orders from an address that does not exist",
        "the last bookshop in a small town on its final day of business",
        "a retired detective who is asked to find a missing cat",
        "a teenager who swaps phones with a stranger by accident",
        "a chef who loses her sense of taste a week before opening her restaurant",
        "a family road trip that goes wrong in a small desert town",
        "an astronaut returning to Earth after two years away",
        "a man who has to give a speech at the wedding of his ex-partner",
        "a village where it has not rained for three years",
        "a night-shift nurse and a patient who cannot sleep",
        "a child who is convinced the neighbour is a spy",
        "a busker who plays the same song every day for a reason nobody knows",
        "two old friends meeting again at a funeral",
    ],
    "openings": [
        "The day the river froze, my father finally told me the truth.",
        "Nobody in our street had ever seen the woman at number nine leave her house.",
        "It was the third time that week the dog had brought home a shoe that wasn't ours.",
        "I knew the train was wrong the moment it left the station.",
        "My sister kept a list of every lie I'd ever told her.",
        "The note on the fridge said only: don't open the garage.",
        "By the time the power came back, everything had changed.",
        "We were supposed to be home before dark.",
        "The new teacher had a scar she never talked about.",
        "Every summer, the same red balloon appeared over the lake.",
    ],
    "images": [
        "an empty swimming pool in winter",
        "a grandmother's kitchen at dawn",
        "the last bus home on a rainy night",
        "a field of sunflowers after the harvest",
        "a hospital waiting room at three in the morning",
        "an old coat found in a charity shop",
        "the sea seen from a moving train",
        "a city street just after snowfall",
        "a box of letters in an attic",
        "a bonfire on the last night of summer",
        "a broken clock in a train station",
        "moving out of a childhood home",
    ],
    "experiences": [
        "the first time you lived away from home",
        "a teacher who changed how you saw yourself",
        "learning to drive",
        "a job you hated and what it taught you",
        "losing a pet",
        "a trip that did not go to plan",
        "the day you realised your parents were ordinary people",
        "moving to a new country",
        "a friendship that ended",
        "running your first race",
        "a time you were completely lost",
        "learning to cook a family recipe",
        "your first concert",
        "failing an important exam",
    ],
    "characters": [
        "a grumpy landlord and a new tenant",
        "two siblings clearing out their late father's flat",
        "a nervous job candidate and a bored interviewer",
        "a teenager and her grandfather",
        "two coworkers on a night shift",
        "a tour guide and a tourist who knows more than she does",
        "a mechanic and a customer who cannot pay",
        "two neighbours arguing over a fence",
    ],
    "situations": [
        "who are waiting for news at a hospital",
        "who are stuck at an airport overnight",
        "on the morning after a big argument",
        "while they are fixing a broken-down car",
        "at a wedding where neither of them knows anyone",
        "during a power cut",
    ],
    # technical
    "howto_tasks": [
        "setting up SSH keys for GitHub",
        "backing up a laptop with an external drive",
        "writing a simple bash script to rename files in bulk",
        "configuring a home Wi-Fi router securely",
        "setting up a Python virtual environment",
        "using Git branches for a small team project",
        "creating a cron job that runs every night",
        "installing and configuring a Raspberry Pi as a home media server",
        "migrating a small website to HTTPS with Let's Encrypt",
        "debugging a slow SQL query",
        "writing unit tests with pytest",
        "setting up two-factor authentication for a small office",
        "building a Docker image for a Node.js app",
        "cleaning up disk space on a Linux server",
        "replacing a laptop battery safely",
        "making a spreadsheet that tracks monthly expenses",
    ],
    "doc_subjects": [
        "the configuration file of a command-line backup tool",
        "the REST endpoints of a small to-do list API",
        "the installation steps for an open-source note-taking app",
        "the command-line flags of an image-resizing utility",
        "the authentication flow of a mobile banking SDK",
        "the plugin system of a static site generator",
        "the environment variables a web service reads at start-up",
        "the data export feature of a fitness-tracking app",
        "the rate limits and error codes of a weather API",
        "the contribution guidelines for an open-source library",
        "the deployment process of a Django application",
        "the caching options of an HTTP client library",
    ],
    "tech_concepts": [
        "public-key cryptography",
        "DNS resolution",
        "garbage collection in a managed language",
        "how a hash table handles collisions",
        "TCP's three-way handshake",
        "database indexing",
        "how HTTPS certificates are verified",
        "containerisation and how it differs from virtual machines",
        "eventual consistency in distributed databases",
        "how a compiler turns source code into machine code",
        "load balancing",
        "how solid-state drives store data",
        "OAuth 2.0",
        "how recommendation algorithms work",
        "the difference between processes and threads",
    ],
    "tech_problems": [
        "a laptop that will not connect to Wi-Fi",
        "a web app that returns a 502 Bad Gateway error",
        "a Python script that fails with ModuleNotFoundError",
        "a printer that shows as offline",
        "a Git merge conflict",
        "a phone that drains its battery overnight",
        "a website that loads slowly on mobile",
        "a database connection that keeps timing out",
        "an email that keeps landing in recipients' spam folders",
        "a Docker container that exits immediately after starting",
        "a smart TV that will not update its apps",
        "a CSV file that opens with garbled characters",
    ],
    # casual
    "forum_situations": [
        "a roommate who never cleans up",
        "choosing between two job offers",
        "a car that makes a weird noise when braking",
        "whether to adopt a second cat",
        "getting back into running after an injury",
        "a neighbour who plays loud music late at night",
        "planning a cheap two-week trip through Europe",
        "starting to learn the guitar at forty",
        "a friend who always cancels plans at the last minute",
        "keeping houseplants alive in a dark flat",
        "asking for a raise for the first time",
        "a teenager who will not get off their phone",
        "switching careers into nursing",
        "dealing with burnout at a new job",
        "building a gaming PC on a budget",
        "a sourdough starter that will not rise",
    ],
    "forum_questions": [
        "the best way to learn a language as an adult",
        "whether an electric car makes sense for a long commute",
        "how to stop procrastinating",
        "whether a standing desk is worth it",
        "how to save money on groceries",
        "the best beginner camera",
        "how to deal with a micromanaging boss",
        "whether to rent or buy a home",
        "getting a toddler to sleep through the night",
        "learning to code without a degree",
        "what to do on a rainy weekend in a small town",
        "how to make friends after moving to a new city",
    ],
    "email_recipients": [
        "your manager",
        "a professor",
        "your landlord",
        "a group of old school friends",
        "a coworker you have not met yet",
        "your neighbour",
        "a customer-service team",
        "your sister",
    ],
    "email_topics": [
        "needing to reschedule a meeting",
        "a problem with the heating in your flat",
        "an extension on a deadline",
        "organising a reunion",
        "a delayed refund",
        "covering a shift next week",
        "a noisy party you are planning",
        "feedback on a project you worked on together",
        "a parcel that was delivered to the wrong address",
        "taking a few days off for a family event",
    ],
    "social_topics": [
        "finally finishing a marathon",
        "a terrible customer-service experience",
        "the first snow of the year",
        "getting a new puppy",
        "a local café closing down",
        "the season finale of a TV show",
        "starting a new job",
        "a weekend camping trip",
        "a home renovation disaster",
        "voting for the first time",
        "a concert you just got back from",
        "learning to bake bread",
    ],
    # review
    "products": [
        "a pair of wireless earbuds",
        "a budget robot vacuum cleaner",
        "an electric kettle",
        "a mid-range Android phone",
        "a mechanical keyboard",
        "a children's balance bike",
        "an air fryer",
        "a hiking backpack",
        "a fitness tracker watch",
        "a cordless drill",
        "a memory-foam mattress",
        "a coffee grinder",
        "a waterproof jacket",
        "a board game for families",
    ],
    "places": [
        "a family-run Italian restaurant",
        "a budget hotel near a train station",
        "a small-town dentist's practice",
        "a seaside campsite",
        "a new climbing gym",
        "a ramen bar",
        "a car repair garage",
        "a museum of local history",
        "a weekend farmers' market",
        "a hair salon",
        "a vegan bakery",
        "a holiday cottage in the countryside",
    ],
    # news
    "news_events": [
        "a town council vote to turn a car park into a public park",
        "a local school winning a national robotics competition",
        "flooding that closed the main road into a market town",
        "a regional hospital opening a new maternity ward",
        "a bus drivers' strike over shift patterns",
        "a rare bird spotted at a city reservoir",
        "a community library that reopened after a fundraising campaign",
        "a new cycle lane that divided local shopkeepers",
        "a power outage that hit thousands of homes during a heatwave",
        "a former factory being converted into affordable housing",
        "a local football club saved from bankruptcy by its fans",
        "a council decision to cut late-night bus services",
    ],
    "op_ed_stances": [
        "that public libraries matter more now than ever",
        "that cities should stop building new car parks",
        "that school days should start later for teenagers",
        "that tipping culture has gone too far",
        "that local newspapers deserve public funding",
        "that remote workers should pay less tax to the cities they left",
        "that smartphones should be banned in schools",
        "that the high street can still be saved",
        "that every child should learn to swim at school",
        "that daylight saving time should be scrapped",
    ],
}

# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

PROMPT_TEMPLATES: list[dict] = [
    # ---- academic -------------------------------------------------------- #
    {
        "id": "acad-essay",
        "domain": "academic",
        "subdomain": "essay",
        "template": "Write an argumentative essay on the question: {question}?",
        "slots": {"question": "essay_questions"},
        "lengths": (300, 450, 600),
    },
    {
        "id": "acad-course-essay",
        "domain": "academic",
        "subdomain": "essay",
        "template": "Write a short essay for a {course} course on {topic}.",
        "slots": {"course": "courses", "topic": "academic_topics"},
        "lengths": (300, 450, 600),
    },
    {
        "id": "acad-abstract",
        "domain": "academic",
        "subdomain": "abstract",
        "template": "Write the abstract of {study}.",
        "slots": {"study": "research_topics"},
        "lengths": (150, 200, 300),
    },
    {
        "id": "acad-explanation",
        "domain": "academic",
        "subdomain": "explanation",
        "template": "Explain {concept} to {audience}.",
        "slots": {"concept": "academic_concepts", "audience": "academic_audiences"},
        "lengths": (150, 200, 300, 450),
    },
    {
        "id": "acad-reflection",
        "domain": "academic",
        "subdomain": "essay",
        "template": (
            "Write a reflective response paper for a {course} class about this week's "
            "reading on {topic}."
        ),
        "slots": {"course": "courses", "topic": "academic_topics"},
        "lengths": (200, 300, 450),
    },
    # ---- creative -------------------------------------------------------- #
    {
        "id": "crea-story",
        "domain": "creative",
        "subdomain": "short_story",
        "template": "Write a short story about {premise}.",
        "slots": {"premise": "story_premises"},
        "lengths": (300, 450, 600),
    },
    {
        "id": "crea-flash",
        "domain": "creative",
        "subdomain": "short_story",
        "template": 'Write a piece of flash fiction that begins with the line: "{opening}"',
        "slots": {"opening": "openings"},
        "lengths": (150, 300, 450),
    },
    {
        "id": "crea-prose-poem",
        "domain": "creative",
        "subdomain": "prose_poem",
        "template": "Write a prose poem, in paragraphs rather than verse lines, about {image}.",
        "slots": {"image": "images"},
        "lengths": (100, 150, 250),
    },
    {
        "id": "crea-narrative",
        "domain": "creative",
        "subdomain": "personal_narrative",
        "template": "Write a first-person personal narrative about {experience}.",
        "slots": {"experience": "experiences"},
        "lengths": (300, 450, 600),
    },
    {
        "id": "crea-scene",
        "domain": "creative",
        "subdomain": "short_story",
        "template": "Write a scene, mostly dialogue, between {characters} {situation}.",
        "slots": {"characters": "characters", "situation": "situations"},
        "lengths": (200, 300, 450),
    },
    # ---- technical ------------------------------------------------------- #
    {
        "id": "tech-howto",
        "domain": "technical",
        "subdomain": "how_to",
        "template": "Write a step-by-step guide to {task}.",
        "slots": {"task": "howto_tasks"},
        "lengths": (200, 300, 450, 600),
    },
    {
        "id": "tech-docs",
        "domain": "technical",
        "subdomain": "documentation",
        "template": "Write the documentation section that describes {subject}.",
        "slots": {"subject": "doc_subjects"},
        "lengths": (200, 300, 450),
    },
    {
        "id": "tech-explainer",
        "domain": "technical",
        "subdomain": "explainer",
        "template": "Write an explainer on {concept} for a developer blog.",
        "slots": {"concept": "tech_concepts"},
        "lengths": (300, 450, 600),
    },
    {
        "id": "tech-troubleshoot",
        "domain": "technical",
        "subdomain": "how_to",
        "template": "Write a troubleshooting guide for {problem}.",
        "slots": {"problem": "tech_problems"},
        "lengths": (200, 300, 450),
    },
    # ---- casual ---------------------------------------------------------- #
    {
        "id": "casu-forum-ask",
        "domain": "casual",
        "subdomain": "forum_post",
        "template": "Write a forum post asking other users for advice about {situation}.",
        "slots": {"situation": "forum_situations"},
        "lengths": (100, 150, 200, 300),
    },
    {
        "id": "casu-forum-reply",
        "domain": "casual",
        "subdomain": "forum_post",
        "template": (
            "Write a reply in a forum thread where someone asked about {question}. "
            "Draw on your own experience."
        ),
        "slots": {"question": "forum_questions"},
        "lengths": (100, 150, 200, 300),
    },
    {
        "id": "casu-email",
        "domain": "casual",
        "subdomain": "email",
        "template": "Write an email to {recipient} about {topic}.",
        "slots": {"recipient": "email_recipients", "topic": "email_topics"},
        "lengths": (100, 150, 200, 300),
    },
    {
        "id": "casu-social",
        "domain": "casual",
        "subdomain": "social_post",
        "template": "Write a social media post about {topic}.",
        "slots": {"topic": "social_topics"},
        "lengths": (100, 150),
    },
    # ---- review ---------------------------------------------------------- #
    {
        "id": "revw-product",
        "domain": "review",
        "subdomain": "product_review",
        "template": "Write a customer review of {product} you bought a few months ago.",
        "slots": {"product": "products"},
        "lengths": (100, 150, 200, 300),
    },
    {
        "id": "revw-place",
        "domain": "review",
        "subdomain": "place_review",
        "template": "Write an online review of {place} you visited recently.",
        "slots": {"place": "places"},
        "lengths": (100, 150, 200, 300),
    },
    # ---- news ------------------------------------------------------------ #
    {
        "id": "news-report",
        "domain": "news",
        "subdomain": "news_report",
        "template": "Write a local news article about {event}. Invent plausible names and quotes.",
        "slots": {"event": "news_events"},
        "lengths": (200, 300, 450),
    },
    {
        "id": "news-oped",
        "domain": "news",
        "subdomain": "opinion",
        "template": "Write a newspaper opinion column arguing {stance}.",
        "slots": {"stance": "op_ed_stances"},
        "lengths": (300, 450, 600),
    },
]

# --------------------------------------------------------------------------- #
# Prompting styles
# --------------------------------------------------------------------------- #

_HUMANLIKE = (
    "Write naturally, the way a real person would, and avoid sounding like an AI: "
    "no generic opening line, no tidy summary at the end, no lists unless they are "
    "really needed."
)

STYLES: list[dict] = [
    {"id": "plain", "kind": "plain", "text": ""},
    {"id": "plain-direct", "kind": "plain", "text": "Keep the writing clear and direct."},
    {
        "id": "persona-student19",
        "kind": "persona",
        "text": "Write it as a 19-year-old college student would.",
    },
    {
        "id": "persona-teacher",
        "kind": "persona",
        "text": "Write it as a high-school teacher in her forties would.",
    },
    {
        "id": "persona-esl",
        "kind": "persona",
        "text": "Write it as a fluent but non-native speaker of English would.",
    },
    {
        "id": "persona-engineer",
        "kind": "persona",
        "text": "Write it as a busy software engineer typing quickly would.",
    },
    {
        "id": "persona-retiree",
        "kind": "persona",
        "text": "Write it as a retired nurse in her sixties would.",
    },
    {"id": "human-natural", "kind": "humanlike", "text": _HUMANLIKE},
    {
        "id": "human-casual",
        "kind": "humanlike",
        "text": (
            "Make it sound like a real person wrote it, not an AI: vary the sentence "
            "length, use contractions, and don't worry about being polished."
        ),
    },
    {
        "id": "human-student19",
        "kind": "persona+humanlike",
        "text": "Write it as a 19-year-old college student would. " + _HUMANLIKE,
    },
]

_TEMPLATES_BY_ID = {t["id"]: t for t in PROMPT_TEMPLATES}
_STYLES_BY_ID = {s["id"]: s for s in STYLES}


# --------------------------------------------------------------------------- #
# Prompt specs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PromptSpec:
    """One concrete prompt: filled template + style + word target + temperature."""

    prompt_id: str
    template_id: str
    domain: str
    subdomain: str
    style: str
    text: str
    target_words: int
    temperature: float
    max_tokens: int
    sample_idx: int = 0


def max_tokens_for(target_words: int) -> int:
    """Generous output-token cap for a word target (~1.3 tokens/word, plus slack)."""
    return int(target_words * 2) + 256


def template_lengths(template: dict) -> tuple[int, ...]:
    return tuple(template.get("lengths") or DEFAULT_LENGTHS)


def build_prompt(
    template_id: str,
    slot_indices: dict[str, int],
    style_id: str,
    target_words: int,
    temperature: float,
    sample_idx: int = 0,
) -> PromptSpec:
    """Build the `PromptSpec` for one fully specified combination."""
    template = _TEMPLATES_BY_ID[template_id]
    style = _STYLES_BY_ID[style_id]
    values = {
        name: SLOT_VALUES[key][slot_indices[name]] for name, key in template["slots"].items()
    }
    parts = [template["template"].format(**values)]
    if style["text"]:
        parts.append(style["text"])
    parts.append(f"Aim for about {target_words} words.")
    parts.append(OUTPUT_INSTRUCTION)

    slot_code = "-".join(f"{name[:3]}{slot_indices[name]:02d}" for name in template["slots"])
    temp_code = f"t{round(temperature * 10):02d}"
    prompt_id = f"{template_id}.{slot_code}.{style_id}.w{target_words}.{temp_code}"
    return PromptSpec(
        prompt_id=prompt_id,
        template_id=template_id,
        domain=template["domain"],
        subdomain=template["subdomain"],
        style=style_id,
        text=" ".join(parts),
        target_words=target_words,
        temperature=temperature,
        max_tokens=max_tokens_for(target_words),
        sample_idx=sample_idx,
    )


def _draw(rng: random.Random, domain: str) -> tuple:
    template = rng.choice([t for t in PROMPT_TEMPLATES if t["domain"] == domain])
    slot_indices = {
        name: rng.randrange(len(SLOT_VALUES[key])) for name, key in template["slots"].items()
    }
    style_id = rng.choice(STYLES)["id"]
    words = rng.choice(template_lengths(template))
    temperature = rng.choice(TEMPERATURES)
    return template["id"], slot_indices, style_id, words, temperature


def sample_prompts(
    n: int, seed: int = 0, domains: Iterable[str] | None = None, max_redraws: int = 50
) -> list[PromptSpec]:
    """Deterministically sample `n` prompts.

    Domains are drawn in shuffled balanced blocks (see `DOMAIN_BLOCK`), so any
    plan of 10 or more prompts covers every requested domain. Within a domain
    the template, slot values, style, word target and temperature are drawn
    at random; a combination already in the plan is redrawn up to
    `max_redraws` times, and only then repeated with the next `sample_idx`.
    The same `(n, seed, domains)` always returns the same list, and a
    smaller `n` with the same seed is a prefix of a larger one.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    wanted = tuple(domains) if domains is not None else tuple(dict.fromkeys(DOMAIN_BLOCK))
    available = {t["domain"] for t in PROMPT_TEMPLATES}
    unknown = [d for d in wanted if d not in available]
    if unknown or not wanted:
        raise ValueError(f"no prompt templates for domains {unknown or list(wanted)}; "
                         f"available: {sorted(available)}")
    block = [d for d in DOMAIN_BLOCK if d in wanted] or list(wanted)

    rng = random.Random(seed)
    seen: Counter[str] = Counter()
    queue: list[str] = []
    specs: list[PromptSpec] = []
    while len(specs) < n:
        if not queue:
            queue = list(block)
            rng.shuffle(queue)
        domain = queue.pop()
        choice = _draw(rng, domain)
        spec = build_prompt(*choice)
        for _ in range(max_redraws):
            if spec.prompt_id not in seen:
                break
            choice = _draw(rng, domain)
            spec = build_prompt(*choice)
        idx = seen[spec.prompt_id]
        seen[spec.prompt_id] += 1
        specs.append(build_prompt(*choice, sample_idx=idx) if idx else spec)
    return specs


def count_combinations() -> int:
    """Number of distinct prompts the bank can produce."""
    total = 0
    for t in PROMPT_TEMPLATES:
        slots = prod(len(SLOT_VALUES[key]) for key in t["slots"].values())
        total += slots * len(STYLES) * len(template_lengths(t)) * len(TEMPERATURES)
    return total


def validate_bank() -> None:
    """Raise ValueError if the bank is inconsistent (used by the tests)."""
    ids = [t["id"] for t in PROMPT_TEMPLATES]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate template ids")
    for t in PROMPT_TEMPLATES:
        if set(t) - {"id", "domain", "subdomain", "template", "slots", "lengths"}:
            raise ValueError(f"{t['id']}: unexpected keys")
        if t["domain"] not in DOMAINS:
            raise ValueError(f"{t['id']}: unknown domain {t['domain']!r}")
        if "." in t["id"] or ":" in t["id"]:
            raise ValueError(f"{t['id']}: ids must not contain '.' or ':'")
        fields = {f for _, f, _, _ in string.Formatter().parse(t["template"]) if f}
        if fields != set(t["slots"]):
            raise ValueError(f"{t['id']}: placeholders {fields} != slots {set(t['slots'])}")
        for key in t["slots"].values():
            if not SLOT_VALUES.get(key):
                raise ValueError(f"{t['id']}: empty or missing slot list {key!r}")
        if len({name[:3] for name in t["slots"]}) != len(t["slots"]):
            raise ValueError(f"{t['id']}: slot names must differ in their first 3 letters")
    for key, values in SLOT_VALUES.items():
        if len(values) > 100:
            raise ValueError(f"slot list {key!r} is too long for 2-digit ids")
