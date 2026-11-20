"""
skolfie/data/generation/essay_plan.py
-------------------------------------
A seeded plan of long (800-1,500 word) student assignments for the Gemini
Batch API run (`scripts/gemini_batch_essays.py`, added 2026-11-20).

Each assignment crosses four things:

- a **genre**: argumentative essay, debate article, persuasive letter, speech,
  compare/contrast, cause and effect, explanatory essay, research paper, lab
  report, book report, literary analysis, critique / rhetorical analysis,
  personal narrative, reflection;
- a **topic** from the pool that fits the genre (debate questions, science and
  society, lab experiments, commonly taught books, life experiences, speeches
  and films, pairs to compare);
- a **grade level** (grade 8 to upper-level university; lab reports, research
  papers and analyses start at grade 9);
- a **student voice** (persona) and, sometimes, a "make it sound like a real
  student" request, as students actually add.

No genre+topic pair appears twice in a plan. Genres are taken in turn
(round robin, seeded order), so each genre gets about the same share until its
pool runs out. The 15 PERSUADE 2.0 assignments come first in every genre that
fits them, because they are the topics the human student essays in the
corpus cover (a topic only AI essays have could become a shortcut).

Prompt ids follow the corpus grouping rules (`skolfie.eval.splits.group_key`):
- PERSUADE topics: `persuade-<name>.<grade>.<genre>-<voice>-<style>.w<N>.n<i>`,
  grouped with the other PERSUADE essays as `fresh:persuade-<name>.<grade>`
  (a `shared_topic_groups` prefix).
- other topics: `gb.<topic>.<genre>-<grade>-<voice>-<style>.w<N>.n<i>`,
  grouped by topic as `fresh:gb.<topic>`, so two genres on one book never
  land on both sides of the split.

`SYSTEM_INSTRUCTION` is the text every request shares (the context-caching
candidate); `PromptSpec.text` is the per-essay part.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from skolfie.data.generation.prompts import PromptSpec
from skolfie.data.generation.student import HUMANIZE

MIN_WORDS = 800
MAX_WORDS = 1500
TARGET_WORDS = tuple(range(850, 1451, 50))   # asked-for length; 850..1450

SYSTEM_INSTRUCTION = f"""\
You write school and university assignments exactly as a real student would hand them in.

Rules for every answer:
- Reply with the assignment text only. No preface ("Sure, here is..."), no notes, comments or \
offers after it, and never mention that you are an AI.
- Plain text only, as typed into a word processor: no Markdown, no asterisks, no pound signs, \
no bold or italics, no tables, no emoji. Separate paragraphs with one blank line.
- Do not put a title at the top, unless the genre is a lab report or research paper.
- Lab reports and research papers may use short plain section headings (for example \
Introduction, Method, Results, Discussion, Conclusion), each on its own line, without numbers \
or symbols. Other genres use ordinary paragraphs, no headings. Avoid bullet lists; if a lab \
report needs a list of materials or steps, use plain lines starting with a hyphen.
- Length matters: write close to the number of words asked for, and never fewer than \
{MIN_WORDS} or more than {MAX_WORDS} words.
- Write at the level and in the voice described: vocabulary, sentence length, structure and \
typical mistakes should fit that student. Invent believable personal details, data and \
examples where the assignment needs them. Mention sources the way that student would \
(for example "according to an article we read in class"); do not invent exact page numbers \
or URLs.
"""


@dataclass(frozen=True)
class Genre:
    id: str
    label: str            # "an argumentative essay"
    pool: tuple[str, ...]  # topic pools it draws from
    min_grade: int = 8     # index into GRADES order: 8..12, then college
    domain: str = "academic"
    subdomain: str = "student_essay"


GENRES: tuple[Genre, ...] = (
    Genre("argumentative", "an argumentative essay that takes a clear position on", ("debate",)),
    Genre("debate-article", "a debate article (an opinion piece for the school newspaper or "
          "website that argues one side) on", ("debate",)),
    Genre("persuasive-letter", "a persuasive letter to someone who can make a decision (a "
          "principal, a city council member or a senator) about", ("debate",)),
    Genre("speech", "a persuasive speech, written out in full, to give to your class or a "
          "school assembly about", ("debate",)),
    Genre("compare-contrast", "a compare-and-contrast essay about", ("compare",)),
    Genre("cause-effect", "a cause-and-effect essay about", ("society",)),
    Genre("explanatory", "an explanatory (informative) essay about", ("society",)),
    Genre("research-paper", "a short research paper, with a thesis and evidence from sources "
          "you name in the text, about", ("society", "debate"), min_grade=9),
    Genre("lab-report", "a lab report (purpose, hypothesis, materials, method, results with "
          "numbers written in the text, discussion, conclusion) on the experiment",
          ("labs",), min_grade=9),
    Genre("book-report", "a book report (summary, characters, themes and your opinion) on",
          ("books",)),
    Genre("literary-analysis", "a literary analysis essay with a thesis about a theme, "
          "character or symbol in", ("books",), min_grade=9),
    Genre("critique", "a critique (rhetorical or critical analysis: what it argues, how, "
          "and how well) of", ("critique",), min_grade=9),
    Genre("narrative", "a personal narrative essay about", ("personal",),
          domain="creative", subdomain="personal_narrative"),
    Genre("reflection", "a reflective essay on what you learned from", ("personal", "reflect"),
          domain="creative", subdomain="personal_narrative"),
)

# Grade id, how the prompt names the writer, numeric level (for min_grade).
GRADES: tuple[tuple[str, str, int], ...] = (
    ("grade8", "an 8th-grade student in the United States", 8),
    ("grade9", "a 9th-grade student in the United States", 9),
    ("grade10", "a 10th-grade student in the United States", 10),
    ("grade11", "an 11th-grade student in the United States", 11),
    ("grade12", "a 12th-grade student in the United States", 12),
    ("college1", "a first-year college student", 13),
    ("college2", "a second-year college student", 14),
    ("college-upper", "a third- or fourth-year university student", 15),
)

VOICES: tuple[tuple[str, str], ...] = (
    ("confident", "You are confident and opinionated and like making bold points."),
    ("anxious", "You are careful and a bit anxious about getting it right, so you over-explain "
                "some points."),
    ("reluctant", "You find writing a chore and are doing this because it is due, so you keep "
                  "things plain and get to the point."),
    ("enthusiast", "You genuinely love this subject and it shows."),
    ("esl", "English is your second language: you write fluently but with a few non-native "
            "phrasings and small grammar slips."),
    ("athlete", "You play a lot of sports and often reach for examples from training and "
                "games."),
    ("gamer", "You spend much of your free time gaming and online, and it shows in your "
              "examples."),
    ("rural", "You grew up in a small rural town and draw on that life."),
    ("city", "You live in a big city and draw on that life."),
    ("first-gen", "You are the first in your family to aim for college and you work hard at "
                  "sounding academic."),
    ("dry-humor", "You have a dry sense of humor that slips into your writing now and then."),
    ("formal", "You try hard to sound formal and academic, sometimes a little too hard."),
    ("part-time-job", "You work a part-time job next to school and write efficiently."),
    ("debater", "You are on the debate team and like structure: claims, evidence, rebuttals."),
)

# PERSUADE 2.0 assignments as topics (name -> topic phrase); same names as
# skolfie.data.generation.student.PERSUADE_PROMPTS.
PERSUADE_TOPICS: dict[str, str] = {
    "phones-and-driving": "whether drivers should be allowed to use cell phones in any way "
                          "while driving",
    "car-free-cities": "limiting car use in cities (car-free neighborhoods in Germany, driving "
                       "bans in Paris, car-free days in Bogota)",
    "summer-projects": "whether school summer projects should be teacher-designed or "
                       "student-designed",
    "cowboy-who-rode-the-waves": "joining the Seagoing Cowboys program after World War II, as "
                                 "Luke Bomberger did",
    "mandatory-extracurricular-activities": "requiring every student to take part in at least "
                                            "one extracurricular activity",
    "exploring-venus": "whether studying Venus is worth it despite the dangers",
    "facial-action-coding-system": "using the Facial Action Coding System to read students' "
                                   "emotions in the classroom",
    "face-on-mars": "the Face on Mars: a natural landform, not something built by aliens",
    "community-service": "whether students should be required to do community service",
    "grades-for-extracurricular-activities": "requiring at least a B average to take part in "
                                             "sports and other activities",
    "driverless-cars": "the development of driverless cars",
    "electoral-college": "keeping the Electoral College or electing the president by popular "
                         "vote",
    "cell-phones-at-school": "whether students should be allowed to use phones at school during "
                             "lunch and free periods",
    "distance-learning": "whether students benefit from attending classes from home online",
    "seeking-multiple-opinions": "why asking more than one person for advice leads to better "
                                 "choices",
}
# Which PERSUADE topics fit the non-debate pools too.
PERSUADE_IN_POOL: dict[str, tuple[str, ...]] = {
    "debate": tuple(PERSUADE_TOPICS),
    "society": ("car-free-cities", "exploring-venus", "face-on-mars", "driverless-cars",
                "electoral-college", "distance-learning", "facial-action-coding-system",
                "seeking-multiple-opinions"),
}


def _lines(block: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in block.strip().splitlines() if s.strip())


POOLS: dict[str, tuple[str, ...]] = {
    "debate": _lines("""
        requiring school uniforms
        banning homework in middle school
        switching to a four-day school week
        starting high school later in the morning
        year-round school calendars
        using SAT and ACT scores in college admissions
        a minimum age of 16 for social media accounts
        banning smartphones for children under 14
        paying college athletes
        making public college tuition-free
        lowering the voting age to 16
        making voting mandatory
        testing cosmetics on animals
        whether zoos should still exist
        banning single-use plastic bags
        nuclear power as a solution to climate change
        a carbon tax on fuel
        serving only vegetarian school lunches on some days
        banning junk food sales in schools
        a tax on sugary drinks
        whether violent video games make teens aggressive
        making esports an official school sport
        letting students use AI chatbots for homework
        whether AI-generated art is real art
        replacing truck drivers with self-driving trucks
        spending on space exploration while Earth has problems
        a universal basic income
        raising the minimum wage
        abolishing daylight saving time
        teaching cursive handwriting in schools
        a required personal finance class in high school
        requiring every student to learn a second language
        cutting arts programs to fund STEM
        taking a gap year before college
        pass/fail grading instead of letter grades
        participation trophies in youth sports
        removing challenged books from school libraries
        teaching coding in elementary school
        security cameras in classrooms
        metal detectors at school entrances
        police officers stationed in schools
        single-sex schools
        replacing printed textbooks with tablets
        banning TikTok
        influencers advertising to children
        the environmental cost of fast fashion
        limiting tourist numbers in popular cities
        gene editing of human embryos
        an opt-out system for organ donation
        raising the legal driving age to 18
        curfews for teenagers
        making public transport free
        replacing street parking with bike lanes
        letting employees work from home permanently
        a four-day work week
        genetically modified food
        lab-grown meat
        reintroducing wolves to national parks
        banning fireworks for private use
        celebrities as role models
        tackle football for kids under 14
        hosting the Olympic Games
        open-book exams
        honor codes and cheating policies
        smaller class sizes
        merit pay for teachers
        recess for high school students
        strict dress codes
        screen time limits for young children
        police use of facial recognition
        online anonymity
        a right to repair your own devices
        banning new gasoline cars by 2035
        limits on short flights
        space tourism
        sending humans to live on Mars
        museums returning artifacts to their countries of origin
        removing statues of controversial historical figures
        free tutoring for every student
        grading students on effort as well as results
        """),
    "compare": _lines("""
        e-books and printed books
        online classes and in-person classes
        city life and country life
        cats and dogs as pets
        public schools and private schools
        texting and talking on the phone
        The Hunger Games novel and its film adaptation
        Romeo and Juliet and West Side Story
        ancient Athens and Sparta
        the Roman Republic and the Roman Empire
        the causes of World War I and World War II
        the American Revolution and the French Revolution
        mitosis and meiosis
        plant cells and animal cells
        renewable and non-renewable energy
        electric cars and gasoline cars
        nuclear power and solar power
        weather and climate
        viruses and bacteria
        the US presidential system and the British parliamentary system
        the poetry of Robert Frost and Emily Dickinson
        the ideas of Martin Luther King Jr. and Malcolm X
        Abraham Lincoln and Franklin D. Roosevelt as wartime leaders
        Thomas Edison and Nikola Tesla
        iPhone and Android phones
        Instagram and TikTok
        streaming services and cable TV
        studying with music and studying in silence
        team sports and individual sports
        a part-time job and volunteering
        living in a dorm and living at home during college
        community college and a four-year university
        schools in the United States and in Finland
        Shakespeare's comedies and tragedies
        Greek and Norse mythology
        Impressionism and Realism in painting
        jazz and classical music
        fast food and home cooking
        the Mediterranean diet and the typical American diet
        hurricanes and tornadoes
        earthquakes and volcanic eruptions
        the Moon and Mars as places for a human base
        the Harry Potter books and films
        1984 and Brave New World
        Lord of the Flies and The Hunger Games
        Mahatma Gandhi and Nelson Mandela
        the Renaissance and the Enlightenment
        the Industrial Revolution and the digital revolution
        handwritten letters and email
        newspapers and social media as news sources
        learning from books and learning from videos
        taking notes by hand and on a laptop
        homeschooling and public school
        working from home and working in an office
        trains and planes for travel
        renting and buying a home
        credit cards and debit cards
        saving money and investing it
        first-person and third-person narration
        your town today and fifty years ago
        Pixar and Studio Ghibli films
        Marvel and DC superhero movies
        chess and strategy video games
        tap water and bottled water
        organic and conventional farming
        zoos and wildlife sanctuaries
        Beethoven and Mozart
        Vincent van Gogh and Pablo Picasso
        ancient Egypt and Mesopotamia
        the Aztec and Inca empires
        the Cold War and today's US-China rivalry
        high school and college
        childhood and the teenage years
        """),
    "society": _lines("""
        climate change
        ocean acidification
        coral reef bleaching
        deforestation in the Amazon
        plastic pollution in the oceans
        microplastics in food and water
        air pollution in big cities
        urban heat islands
        droughts and water shortages
        wildfires
        the decline of bees and other pollinators
        invasive species
        antibiotic resistance
        how vaccines are developed
        the 1918 flu pandemic
        the COVID-19 pandemic and schools
        sleep deprivation in teenagers
        energy drinks and caffeine
        screen time and attention span
        social media and teen mental health
        cyberbullying
        online misinformation
        deepfakes
        artificial intelligence and jobs
        automation in factories
        the gig economy
        inflation
        student loan debt
        rising housing costs
        food waste
        childhood obesity
        concussions in sports
        doping in sports
        the Great Depression
        the Dust Bowl
        the Industrial Revolution
        the printing press
        the Black Death
        the fall of the Western Roman Empire
        the French Revolution
        the American Civil War
        the civil rights movement
        women's suffrage
        the space race
        the fall of the Berlin Wall
        the invention of the internet
        how smartphones changed daily life
        the Silk Road
        the Columbian Exchange
        immigration to the United States
        urbanization
        globalization
        the growth of renewable energy
        the Chernobyl disaster
        the Apollo 11 Moon landing
        the James Webb Space Telescope
        black holes
        the water cycle
        photosynthesis
        plate tectonics
        how volcanoes form
        earthquakes and tsunamis
        how hurricanes form
        the human immune system
        DNA and heredity
        CRISPR gene editing
        stem cell research
        the placebo effect
        how memory works and how to study
        multitasking
        music and the brain
        growing up bilingual
        video games and learning
        homework and school achievement
        exercise and mental health
        nutrition and school performance
        poverty and education
        the digital divide
        young people and voter turnout
        advertising aimed at children
        why recycling plastic is so hard
        endangered species
        the rise of music streaming
        the history of hip-hop
        the Harlem Renaissance
        the Renaissance
        the Scientific Revolution
        Edward Jenner and the first vaccine
        the sinking of the Titanic
        the history of the Olympic Games
        the Great Wall of China
        the Underground Railroad
        the Marshall Plan
        the European Union
        the history of women in science
        the rise of fast food
        light pollution
        noise pollution in cities
        overfishing
        melting glaciers and sea level rise
        electric vehicle batteries and lithium mining
        the history of the bicycle
        the invention of the airplane
        the history of public libraries
        the Cuban Missile Crisis
        the Montgomery bus boycott
        the Lewis and Clark expedition
        the Gold Rush
        the Irish potato famine
        the history of vaccines against polio
        the effects of noise on concentration
        loneliness among young people
        the four-day school week in rural districts
        the history of the minimum wage
        """),
    "labs": _lines("""
        enzyme activity (catalase) at different temperatures
        osmosis in potato slices in salt solutions
        light intensity and photosynthesis in pondweed
        yeast fermentation with different sugars
        pendulum length and period
        acceleration of a cart down a ramp
        Newton's second law with carts and masses
        specific heat capacity of metals
        density of irregular objects
        acid-base titration of vinegar
        temperature and the reaction rate of antacid tablets
        surface area and reaction rate
        conservation of mass in a closed reaction
        flame tests of metal salts
        paper chromatography of marker inks
        electrolysis of water
        building a lemon battery
        Ohm's law with resistors
        magnet strength and distance
        projectile motion with a launcher
        friction on different surfaces
        heart rate before and after exercise
        reaction time with a ruler drop test
        lung capacity measurements
        DNA extraction from strawberries
        hand sanitizer and bacterial growth on agar
        salt concentration and seed germination
        plant growth under colored light
        water quality of a local stream
        soil pH from different locations
        buoyancy and Archimedes' principle
        salt and the boiling point of water
        freezing point depression with salt
        evaporation rates of different liquids
        insulating materials and heat loss
        building a solar oven
        paper airplane design and flight distance
        drop height and bounce height of a ball
        Hooke's law with springs
        measuring the speed of sound
        refraction of light in water and glass
        the focal length of a convex lens
        pH of household substances with red cabbage indicator
        vitamin C content of fruit juices
        diffusion in agar cubes of different sizes
        transpiration in plants
        mitosis in onion root tip cells
        microorganisms in pond water under a microscope
        owl pellet dissection
        earthworm responses to light and moisture
        salinity and brine shrimp hatching
        rusting of iron nails in different conditions
        melting ice with different salts
        growing crystals
        an egg drop challenge
        a density column of liquids
        the cooling curve of stearic acid
        percent yield of a precipitation reaction
        mechanical advantage of pulleys
        the effect of temperature on dissolving sugar
        """),
    "books": _lines("""
        To Kill a Mockingbird
        The Great Gatsby
        Of Mice and Men
        Lord of the Flies
        1984
        Animal Farm
        Fahrenheit 451
        The Catcher in the Rye
        The Giver
        The Outsiders
        Hatchet
        Wonder
        The Hunger Games
        Holes
        Number the Stars
        The Diary of a Young Girl by Anne Frank
        Night by Elie Wiesel
        Romeo and Juliet
        Macbeth
        Hamlet
        Julius Caesar
        A Midsummer Night's Dream
        Othello
        The Crucible
        Death of a Salesman
        A Raisin in the Sun
        Frankenstein
        Pride and Prejudice
        Jane Eyre
        Great Expectations
        A Christmas Carol
        The Odyssey
        Beowulf
        The Scarlet Letter
        Adventures of Huckleberry Finn
        Little Women
        The Call of the Wild
        Treasure Island
        The Old Man and the Sea
        The Pearl
        Their Eyes Were Watching God
        The Kite Runner
        The House on Mango Street
        Speak
        The Book Thief
        The Absolutely True Diary of a Part-Time Indian
        Brown Girl Dreaming
        Esperanza Rising
        Bud, Not Buddy
        Bridge to Terabithia
        Tuck Everlasting
        A Wrinkle in Time
        The Hobbit
        The Lion, the Witch and the Wardrobe
        Harry Potter and the Sorcerer's Stone
        Percy Jackson and the Lightning Thief
        The Maze Runner
        The Fault in Our Stars
        Things Fall Apart
        The Alchemist
        Brave New World
        Slaughterhouse-Five
        The Road
        Never Let Me Go
        Life of Pi
        The Curious Incident of the Dog in the Night-Time
        Persepolis
        Maus
        I Know Why the Caged Bird Sings
        Long Way Down
        The Hate U Give
        Dracula
        The Strange Case of Dr Jekyll and Mr Hyde
        Into the Wild
        Hidden Figures
        Educated
        The Immortal Life of Henrietta Lacks
        Born a Crime
        I Am Malala
        Flowers for Algernon
        """),
    "critique": _lines("""
        Martin Luther King Jr.'s "I Have a Dream" speech
        Abraham Lincoln's Gettysburg Address
        John F. Kennedy's "We choose to go to the Moon" speech
        Malala Yousafzai's speech to the United Nations
        Steve Jobs' 2005 Stanford commencement speech
        Greta Thunberg's 2019 UN climate speech
        Franklin D. Roosevelt's "Day of Infamy" speech
        Sojourner Truth's "Ain't I a Woman?"
        Winston Churchill's "We shall fight on the beaches" speech
        the documentary Super Size Me
        the documentary Blackfish
        the documentary My Octopus Teacher
        the documentary The Social Dilemma
        the documentary An Inconvenient Truth
        the documentary Free Solo
        the film Inside Out
        the film WALL-E
        the film Dead Poets Society
        the film Remember the Titans
        the film The Truman Show
        the film Coco
        the film Spirited Away
        the film Black Panther
        the film Oppenheimer
        the film Jurassic Park
        the film The Breakfast Club
        Ken Robinson's TED talk "Do schools kill creativity?"
        Angela Duckworth's TED talk on grit
        Brene Brown's TED talk on vulnerability
        Simon Sinek's TED talk "Start with why"
        Nicholas Carr's essay "Is Google Making Us Stupid?"
        Brent Staples' essay "Just Walk on By"
        George Orwell's essay "Politics and the English Language"
        Jonathan Swift's "A Modest Proposal"
        Ray Bradbury's story "There Will Come Soft Rains"
        Shirley Jackson's story "The Lottery"
        Kurt Vonnegut's story "Harrison Bergeron"
        Edgar Allan Poe's story "The Tell-Tale Heart"
        O. Henry's story "The Gift of the Magi"
        Robert Frost's poem "The Road Not Taken"
        Langston Hughes' poem "Harlem"
        Maya Angelou's poem "Still I Rise"
        Emily Dickinson's poem "Because I could not stop for Death"
        Charlotte Perkins Gilman's story "The Yellow Wallpaper"
        Frederick Douglass' speech "What to the Slave Is the Fourth of July?"
        the novel-to-film adaptation of The Hunger Games
        a popular advertising campaign of your choice
        your school's mission statement
        a news article about a local issue
        a recent Pixar short film
        """),
    "personal": _lines("""
        your first day at a new school
        moving to a new city
        learning to ride a bike
        a sports game your team lost
        a big game your team won
        a summer job
        cooking a meal for your family for the first time
        getting lost somewhere unfamiliar
        a family trip
        a story your grandparent told you
        losing a pet
        adopting a pet
        a friendship that ended
        making an unexpected new friend
        standing up to a bully
        a mistake that taught you something
        failing an important test
        a teacher who changed your mind about a subject
        learning a musical instrument
        performing in a school play or concert
        a science fair project
        a camping trip
        a storm or a long power outage
        a stay in the hospital
        breaking a bone
        a family wedding
        a religious or cultural holiday in your family
        moving to a new country and learning a new language
        translating for your parents
        babysitting your younger siblings
        a day of volunteering
        organizing a fundraiser
        running for student council
        your first debate tournament
        a long road trip
        a time you were really scared
        a time you felt proud of yourself
        getting your first phone
        taking a month off social media
        a video game tournament
        a hobby you started during the pandemic lockdown
        doing school online during the pandemic
        working at a fast food restaurant
        a coach who pushed you hard
        quitting something you had done for years
        trying out for a team and not making it
        a rivalry with a sibling
        helping a stranger
        splitting time between your parents' homes
        an embarrassing moment
        your first concert
        a museum visit that stuck with you
        a hike in nature
        learning to drive
        a disagreement with your parents
        your name and what it means
        a family recipe
        a place that feels like home
        a group project that went wrong
        tutoring a younger student
        preparing for a big exam
        applying to college
        a college campus visit
        a summer camp
        building something with your own hands
        a robotics competition
        an art project you are proud of
        a book that changed how you think
        a song that means a lot to you
        a week without internet
        losing something important
        meeting someone you admire
        being the new kid
        a cultural misunderstanding
        learning about your family's heritage
        a sports injury
        running your first race
        a dance recital
        having to apologize
        working on a farm
        """),
    "reflect": _lines("""
        your growth as a writer this semester
        a class you just finished
        an internship
        a semester-long group project
        a service-learning course
        your first year of college
        your first year of high school
        a study-abroad or exchange experience
        a mock trial or Model UN conference
        a peer review of your own essay
        a failed experiment in a science class
        learning to manage your time
        a leadership role in a club
        a long research project
        a public speaking assignment
        """),
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str, max_len: int = 48) -> str:
    out = _SLUG_RE.sub("-", text.lower()).strip("-")
    return out[:max_len].rstrip("-") or "x"


@dataclass(frozen=True)
class _Pair:
    genre: Genre
    topic_key: str     # "persuade-<name>" or the topic slug
    topic_text: str
    persuade: bool


def _pairs_by_genre(rng: random.Random) -> dict[str, list[_Pair]]:
    """For each genre, its (genre, topic) pairs: PERSUADE topics first (shuffled), then the
    others (shuffled)."""
    out: dict[str, list[_Pair]] = {}
    for genre in GENRES:
        persuade, other, seen = [], [], set()
        for pool in genre.pool:
            for name in PERSUADE_IN_POOL.get(pool, ()):
                key = f"persuade-{name}"
                if key not in seen:
                    seen.add(key)
                    persuade.append(_Pair(genre, key, PERSUADE_TOPICS[name], True))
            for text in POOLS[pool]:
                key = slug(text)
                if key not in seen:
                    seen.add(key)
                    other.append(_Pair(genre, key, text, False))
        rng.shuffle(persuade)
        rng.shuffle(other)
        out[genre.id] = persuade + other
    return out


def capacity() -> int:
    """How many distinct genre+topic pairs exist (the largest possible plan)."""
    return sum(len(v) for v in _pairs_by_genre(random.Random(0)).values())


def _spec(pair: _Pair, rng: random.Random, idx: int) -> PromptSpec:
    genre = pair.genre
    grades = [g for g in GRADES if g[2] >= genre.min_grade]
    grade_id, grade_text, _ = rng.choice(grades)
    voice_id, voice = rng.choice(VOICES)
    style_id, style = rng.choice(HUMANIZE)
    words = rng.choice(TARGET_WORDS)
    parts = [f"Assignment: write {genre.label} {pair.topic_text}.",
             f"You are {grade_text}. {voice}"]
    if style:
        parts.append(style)
    parts.append(f"Length: about {words} words (it must be between {MIN_WORDS} and "
                 f"{MAX_WORDS} words).")
    tail = f"{genre.id}-{grade_id}-{voice_id}-{style_id}.w{words}.n{idx:04d}"
    if pair.persuade:
        prompt_id = f"{pair.topic_key}.{grade_id}.{tail}"
    else:
        prompt_id = f"gb.{pair.topic_key}.{tail}"
    return PromptSpec(prompt_id=prompt_id, template_id=f"{genre.id}:{pair.topic_key}",
                      domain=genre.domain, subdomain=genre.subdomain,
                      style=f"{grade_id}+{voice_id}+{style_id}", text=" ".join(parts),
                      target_words=words, temperature=1.0, max_tokens=4096)


def essay_plan(n: int, seed: int = 0) -> list[PromptSpec]:
    """`n` assignments with distinct genre+topic pairs, genres taken in turn. Deterministic for
    a seed, and a prefix of a longer plan (plan(n) == plan(m)[:n] for n <= m), so a top-up
    round can take the next entries. Raises ValueError if `n` exceeds `capacity()`."""
    if n < 0:
        raise ValueError("n must be >= 0")
    rng = random.Random(seed)
    queues = _pairs_by_genre(rng)
    order = [g.id for g in GENRES]
    rng.shuffle(order)
    pairs: list[_Pair] = []
    while len(pairs) < n and any(queues[g] for g in order):
        for gid in order:
            if queues[gid] and len(pairs) < n:
                pairs.append(queues[gid].pop(0))
    if len(pairs) < n:
        raise ValueError(f"only {len(pairs)} distinct genre+topic pairs; asked for {n}")
    spec_rng = random.Random(seed + 1)
    return [_spec(p, spec_rng, i) for i, p in enumerate(pairs)]
