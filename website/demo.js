// Skolfie "how it works" demo: a live UI animation in the hero card.
// One running example (Maya's climate essay, class 9B) told in short chapters,
// each with one caption that lights up word by word. No video file, no sound.
// Viewers who ask for reduced motion get a still frame and can press play.
(function () {
  "use strict";
  var fig = document.querySelector(".demo");
  if (!fig) return;
  var stage = fig.querySelector(".js-demo-stage");
  var app = stage.querySelector('[data-scene="app"]');
  var cursor = stage.querySelector(".js-cursor");
  var caption = stage.querySelector(".js-caption");
  var track = fig.querySelector(".js-track");
  var timeEl = fig.querySelector(".js-time");
  var playBtn = fig.querySelector(".js-play");
  var result = stage.querySelector(".js-result");
  var rows = Array.prototype.slice.call(stage.querySelectorAll(".rows li"));
  var countEl = stage.querySelector(".js-count");

  function $(sel) { return stage.querySelector(sel); }

  // ---------- Stage helpers ----------
  var FLAGS = ["is-filled", "is-scanning", "is-verdict", "is-marks", "is-tip", "is-reasons", "is-warn", "is-pressed"];
  function scene(name) {
    Array.prototype.forEach.call(stage.querySelectorAll(".scene"), function (s) {
      s.classList.toggle("is-on", s.getAttribute("data-scene") === name);
    });
  }
  function view(name) {
    app.setAttribute("data-view", name);
    $(".js-seg-one").classList.toggle("is-on", name !== "folder");
    $(".js-seg-folder").classList.toggle("is-on", name === "folder");
  }
  function flags(list) {
    FLAGS.forEach(function (f) { stage.classList.toggle(f, list.indexOf(f) >= 0); });
  }
  function on(f) { stage.classList.add(f); }
  function scrollResult(sel) {
    // Slide the result up so the element is in view above the caption.
    if (!sel) { result.style.setProperty("--scroll", "0px"); return; }
    var el = $(sel);
    var room = stage.clientHeight - 150;
    var need = el.offsetTop + el.offsetHeight - room;
    result.style.setProperty("--scroll", -Math.max(0, need) + "px");
  }
  function folderRows(n) {
    rows.forEach(function (r, i) { r.classList.toggle("is-in", i < n); });
    countEl.textContent = n >= rows.length ? "28" : String(Math.round(n * 28 / rows.length));
  }

  // Cursor: moves to the centre of an element (instantly when jump is true).
  function point(sel, jump, dx, dy) {
    var el = $(sel);
    if (!el || !el.offsetParent) return;
    var s = stage.getBoundingClientRect(), r = el.getBoundingClientRect();
    cursor.classList.toggle("no-anim", !!jump);
    cursor.style.setProperty("--x", (r.left - s.left + r.width * (dx == null ? 0.5 : dx)) + "px");
    cursor.style.setProperty("--y", (r.top - s.top + r.height * (dy == null ? 0.5 : dy)) + "px");
    cursor.classList.add("is-on");
    if (jump) { void cursor.offsetWidth; cursor.classList.remove("no-anim"); }
  }
  function hideCursor() { cursor.classList.remove("is-on"); }
  function click() {
    cursor.classList.remove("is-click"); void cursor.offsetWidth; cursor.classList.add("is-click");
  }

  // ---------- Chapters ----------
  // enter() sets the full state at the chapter's start, so seeking works;
  // events run at ms offsets inside the chapter.
  var CH = [
    { name: "Intro", dur: 3600, cap: "A second reader for every essay.",
      enter: function () { scene("intro"); hideCursor(); } },
    { name: "Paste", dur: 5000, cap: "Paste a student's essay.",
      enter: function () { scene("app"); view("compose"); flags([]); scrollResult(); point(".js-seg-folder", true); },
      events: [[300, function () { point(".js-composer", false, 0.5, 0.35); }],
               [1300, function () { click(); }],
               [1500, function () { on("is-filled"); }],
               [2900, function () { point(".js-check", false); }]] },
    { name: "Verdict", dur: 5000, cap: "Get a clear verdict: Human, AI, Mixed or Uncertain.",
      enter: function () { scene("app"); view("compose"); flags(["is-filled"]); scrollResult(); point(".js-check", true); },
      events: [[250, function () { click(); on("is-pressed"); }],
               [550, function () { view("result"); on("is-scanning"); point(".verdict__value", false, 1.6, 2); }],
               [1700, function () { stage.classList.remove("is-scanning"); on("is-verdict"); }]] },
    { name: "Highlights", dur: 5000, cap: "Mixed essays show which passages look AI-written.",
      enter: function () { scene("app"); view("result"); flags(["is-filled", "is-verdict"]); scrollResult(); point(".verdict__value", true, 1.6, 2); },
      events: [[300, function () { on("is-marks"); }],
               [1500, function () { point(".js-mark", false, 0.35, 0.3); }],
               [2300, function () { on("is-tip"); }]] },
    { name: "Reasons", dur: 5000, cap: "Every verdict comes with reasons you can read.",
      enter: function () { scene("app"); view("result"); flags(["is-filled", "is-verdict", "is-marks"]); scrollResult(); point(".js-mark", true, 0.35, 0.3); },
      events: [[200, function () { scrollResult(".js-reasons"); }],
               [700, function () { on("is-reasons"); point(".js-reasons", false, 0.6, 0.6); }]] },
    { name: "Warnings", dur: 5200, cap: "English learners get flagged more often, so it warns you.",
      enter: function () { scene("app"); view("result"); flags(["is-filled", "is-verdict", "is-marks", "is-reasons"]); scrollResult(".js-reasons"); point(".js-reasons", true, 0.6, 0.6); },
      events: [[200, function () { scrollResult(".js-warn"); }],
               [700, function () { on("is-warn"); }],
               [1200, function () { point(".js-warn", false, 0.5, 0.55); }]] },
    { name: "Class folder", dur: 6000, cap: "Check a whole class folder at once.",
      enter: function () { scene("app"); view("result"); flags(["is-filled", "is-verdict", "is-marks", "is-reasons", "is-warn"]); scrollResult(".js-warn"); folderRows(0); point(".js-warn", true, 0.5, 0.55); },
      events: [[200, function () { point(".js-seg-folder", false); }],
               [1100, function () { click(); view("folder"); }],
               [1500, function () { folderRows(1); }], [1850, function () { folderRows(2); }],
               [2200, function () { folderRows(3); }], [2550, function () { folderRows(4); }],
               [2900, function () { folderRows(5); }], [3250, function () { folderRows(6); }],
               [3700, function () { point(".rows li:nth-child(5) .tag", false); }]] },
    { name: "Accuracy", dur: 5000, cap: "Built to rarely flag honest work.",
      enter: function () { scene("stat"); hideCursor(); } },
    { name: "Privacy", dur: 5000, cap: "Essays are never sent to third-party AI.",
      enter: function () { scene("privacy"); hideCursor(); } },
    { name: "Fair use", dur: 5200, cap: "Use it to start a conversation, not as proof.",
      enter: function () { scene("talk"); hideCursor(); } }
  ];
  var total = 0;
  CH.forEach(function (c) { c.start = total; total += c.dur; c.words = c.cap.split(" "); });

  function fmt(ms) {
    var s = Math.floor(ms / 1000);
    return Math.floor(s / 60) + ":" + ("0" + (s % 60)).slice(-2);
  }

  // Progress bar: one segment per chapter, width by duration; click to jump.
  var segs = CH.map(function (c, i) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "demo__seg";
    b.style.flexGrow = String(c.dur);
    b.setAttribute("aria-label", "Chapter " + (i + 1) + ": " + c.name);
    b.innerHTML = "<span></span>";
    b.addEventListener("click", function () { seek(i); play(); });
    track.appendChild(b);
    return b;
  });

  // Transcript for screen readers.
  var tr = fig.querySelector(".js-transcript");
  CH.forEach(function (c) {
    var li = document.createElement("li");
    li.textContent = c.name + ": " + c.cap;
    tr.appendChild(li);
  });

  // ---------- Playback ----------
  var t = 0, cur = -1, fired = 0, playing = false, last = null, raf = 0, wordsShown = -1;

  function renderCaption(c, local) {
    var per = Math.min(340, (c.dur * 0.72) / c.words.length);
    var n = Math.floor((local - 250) / per) + 1;
    n = Math.max(0, Math.min(c.words.length, n));
    if (n === wordsShown) return;
    wordsShown = n;
    var spans = caption.children;
    for (var i = 0; i < spans.length; i++) spans[i].className = i < n - 1 ? "is-past" : i === n - 1 ? "is-now" : "";
  }

  function enterChapter(i) {
    cur = i; fired = 0; wordsShown = -1;
    stage.classList.remove("is-ended");
    CH[i].enter();
    caption.innerHTML = "";
    CH[i].words.forEach(function (w) {
      var s = document.createElement("span");
      s.textContent = w;
      caption.appendChild(s);
      caption.appendChild(document.createTextNode(" "));
    });
    caption.classList.remove("is-in"); void caption.offsetWidth; caption.classList.add("is-in");
  }

  function update() {
    if (t >= total) { end(); return; }
    var i = cur;
    while (i < CH.length - 1 && t >= CH[i + 1].start) i++;
    if (i < 0) i = 0;
    if (i !== cur) enterChapter(i);
    var c = CH[cur], local = t - c.start, ev = c.events || [];
    while (fired < ev.length && local >= ev[fired][0]) { ev[fired][1](); fired++; }
    renderCaption(c, local);
    segs.forEach(function (s, k) {
      var f = k < cur ? 1 : k > cur ? 0 : local / c.dur;
      s.firstChild.style.transform = "scaleX(" + f.toFixed(4) + ")";
      s.classList.toggle("is-cur", k === cur);
    });
    timeEl.textContent = c.name + " · " + fmt(t) + " / " + fmt(total);
  }

  function frame(ts) {
    if (!playing) return;
    if (last !== null) t += Math.min(100, ts - last);
    last = ts;
    update();
    if (playing) raf = requestAnimationFrame(frame);
  }
  function play() {
    if (playing) return;
    if (t >= total) seek(0);
    playing = true; last = null;
    fig.classList.add("is-playing");
    playBtn.setAttribute("aria-label", "Pause demo");
    raf = requestAnimationFrame(frame);
  }
  function pause() {
    playing = false;
    cancelAnimationFrame(raf);
    fig.classList.remove("is-playing");
    playBtn.setAttribute("aria-label", "Play demo");
  }
  function seek(i) {
    t = CH[i].start; cur = -1;
    // Seeking backwards or forwards: rebuild the chapter's start state.
    update();
  }
  function end() {
    t = total;
    pause();
    stage.classList.add("is-ended");
    segs.forEach(function (s) { s.firstChild.style.transform = "scaleX(1)"; s.classList.remove("is-cur"); });
    timeEl.textContent = "Watch again · " + fmt(total) + " / " + fmt(total);
    hideCursor();
    stage.setAttribute("aria-hidden", "false");
  }

  playBtn.addEventListener("click", function () { if (playing) { userPaused = true; pause(); } else { userPaused = false; play(); } });
  stage.querySelector(".js-replay").addEventListener("click", function () {
    stage.setAttribute("aria-hidden", "true");
    seek(0); play();
    playBtn.focus();
  });
  var userPaused = false;

  // Show the first chapter, then autoplay while the card is on screen (motion allowed only).
  var reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) {
    // A still frame with the verdict, highlights and reasons; play stays available.
    fig.classList.add("is-still");
    seek(4);
    on("is-reasons");
    scrollResult();
    hideCursor();
    caption.innerHTML = "";
    timeEl.textContent = "Press play for the 50-second demo";
  } else {
    seek(0);
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting && e.intersectionRatio > 0.35) { if (!userPaused && t < total) play(); }
          else if (playing) pause();
        });
      }, { threshold: [0, 0.35, 0.6] }).observe(stage);
    }
    document.addEventListener("visibilitychange", function () { if (document.hidden && playing) pause(); });
  }
  // Re-aim the cursor and scroll when the layout changes size.
  window.addEventListener("resize", function () { if (cur >= 0 && t < total) { var keep = t; seek(cur); t = keep; update(); } });
})();
