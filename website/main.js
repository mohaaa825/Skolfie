// Skolfie website behaviour: menu, tabs, and the app link from config.js.
(function () {
  "use strict";
  var cfg = window.SKOLFIE_CONFIG || {};

  function safeHttpUrl(value) {
    if (!value) return "";
    try {
      var u = new URL(String(value), window.location.href);
      return u.protocol === "https:" || u.protocol === "http:" ? u.href : "";
    } catch (e) {
      return "";
    }
  }

  // App link: every .js-app button opens the checker when a link is set.
  var appUrl = safeHttpUrl(cfg.appUrl);
  document.querySelectorAll(".js-app").forEach(function (a) {
    if (appUrl) {
      a.href = appUrl;
      a.target = "_blank";
      a.rel = "noopener";
    } else if (a.classList.contains("js-app-strict")) {
      a.hidden = true;
    }
  });
  var email = String(cfg.contactEmail || "").trim();
  var hasEmail = /^[^\s@<>"]+@[^\s@<>"]+\.[^\s@<>"]+$/.test(email);
  if (!appUrl) {
    var note = document.querySelector(".js-try-note");
    if (note) {
      note.textContent = hasEmail
        ? "The online checker isn't public yet. Ask for access and we'll send your school a link."
        : "The online checker isn't public yet. It opens here soon.";
    }
  }

  document.querySelectorAll(".js-contact").forEach(function (a) {
    if (hasEmail) {
      a.href = "mailto:" + email + "?subject=" + encodeURIComponent("Skolfie for our school");
    } else {
      a.hidden = true;
    }
  });

  var year = document.querySelector(".js-year");
  if (year) year.textContent = String(new Date().getFullYear());

  // Sticky nav border once the page scrolls.
  var nav = document.querySelector(".nav");
  function onScroll() { nav.classList.toggle("is-scrolled", window.scrollY > 8); }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // Mobile menu.
  var toggle = document.querySelector(".nav__toggle");
  var links = document.getElementById("nav-links");
  function setMenu(open) {
    toggle.setAttribute("aria-expanded", String(open));
    links.classList.toggle("is-open", open);
  }
  toggle.addEventListener("click", function () {
    setMenu(toggle.getAttribute("aria-expanded") !== "true");
  });
  links.addEventListener("click", function (e) {
    if (e.target.closest("a")) setMenu(false);
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") setMenu(false);
  });

  // ---------- Motion ----------
  var root = document.documentElement;
  var motion = root.classList.contains("motion");

  // Headline: reveal word by word.
  var words = document.querySelector(".js-words");
  if (motion && words) {
    var label = words.textContent;
    words.setAttribute("aria-label", label);
    words.textContent = "";
    label.split(" ").forEach(function (w, i, all) {
      var span = document.createElement("span");
      span.className = "word";
      span.setAttribute("aria-hidden", "true");
      span.style.setProperty("--i", i);
      span.textContent = w;
      words.appendChild(span);
      if (i < all.length - 1) words.appendChild(document.createTextNode(" "));
    });
  }

  // Count-up numbers.
  function countUp(el) {
    var target = parseFloat(el.getAttribute("data-count"));
    var decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
    if (!motion || isNaN(target)) return;
    var start = null, dur = 1400;
    function step(ts) {
      if (start === null) start = ts;
      var t = Math.min(1, (ts - start) / dur);
      var eased = 1 - Math.pow(1 - t, 3);
      el.textContent = (target * eased).toFixed(decimals);
      if (t < 1) requestAnimationFrame(step);
    }
    el.textContent = (0).toFixed(decimals);
    requestAnimationFrame(step);
  }

  // Scroll reveal, staggered inside each group.
  var revealEls = Array.prototype.slice.call(document.querySelectorAll("[data-reveal]"));
  function reveal(el) {
    if (el.classList.contains("is-in")) return;
    el.classList.add("is-in");
    Array.prototype.forEach.call(el.querySelectorAll("[data-count]"), countUp);
  }
  if (motion && "IntersectionObserver" in window) {
    revealEls.forEach(function (el) {
      var sibs = Array.prototype.filter.call(el.parentNode.children, function (c) { return c.hasAttribute("data-reveal"); });
      var idx = sibs.indexOf(el);
      if (idx > 0) el.style.setProperty("--delay", Math.min(idx, 5) * 0.09 + "s");
    });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { reveal(e.target); io.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.12 });
    revealEls.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add("is-in"); });
  }

  // Gentle parallax on the background photos.
  var layers = Array.prototype.slice.call(document.querySelectorAll(".hero__img, .closing__photo"));
  if (motion && layers.length) {
    var ticking = false;
    function parallax() {
      ticking = false;
      var vh = window.innerHeight;
      layers.forEach(function (el) {
        var r = el.parentNode.getBoundingClientRect();
        if (r.bottom < 0 || r.top > vh) return;
        var offset = (r.top + r.height / 2 - vh / 2) * -0.12;
        el.style.setProperty("--parallax", offset.toFixed(1) + "px");
      });
    }
    window.addEventListener("scroll", function () {
      if (!ticking) { ticking = true; requestAnimationFrame(parallax); }
    }, { passive: true });
    parallax();
  }

  window.__skolfieReady = true;

  // Tabs in "How it works" (arrow keys move between tabs).
  var tabs = Array.prototype.slice.call(document.querySelectorAll('[role="tab"]'));
  function select(tab, focus) {
    tabs.forEach(function (t) {
      var on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
    if (focus) tab.focus();
  }
  tabs.forEach(function (tab, i) {
    tab.addEventListener("click", function () { select(tab, false); });
    tab.addEventListener("keydown", function (e) {
      var next = null;
      if (e.key === "ArrowRight") next = tabs[(i + 1) % tabs.length];
      if (e.key === "ArrowLeft") next = tabs[(i - 1 + tabs.length) % tabs.length];
      if (e.key === "Home") next = tabs[0];
      if (e.key === "End") next = tabs[tabs.length - 1];
      if (next) { e.preventDefault(); select(next, true); }
    });
  });
})();
