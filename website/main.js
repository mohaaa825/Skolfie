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

  var appUrl = safeHttpUrl(cfg.appUrl);
  var email = String(cfg.contactEmail || "").trim();
  var hasEmail = /^[^\s@<>"]+@[^\s@<>"]+\.[^\s@<>"]+$/.test(email);
  var accessMail = hasEmail ? "mailto:" + email + "?subject=" + encodeURIComponent("Skolfie for our school") : "";

  // App buttons say "Request access" until the checker is public. With a link set, they open the
  // checker and switch to their live label; without one, they email us (or scroll to "Try Skolfie").
  document.querySelectorAll(".js-app").forEach(function (a) {
    if (appUrl) {
      a.href = appUrl;
      a.target = "_blank";
      a.rel = "noopener";
      var live = a.getAttribute("data-live-label");
      if (live) a.textContent = live;
    } else if (a.classList.contains("js-app-strict")) {
      a.hidden = true;
    } else if (accessMail) {
      a.href = accessMail;
    }
  });
  if (!appUrl) {
    var note = document.querySelector(".js-try-note");
    if (note) {
      note.textContent = hasEmail
        ? "The online checker isn't public yet. Ask for access and we'll send your school a link."
        : "The online checker isn't public yet. It opens here soon.";
    }
  }

  document.querySelectorAll(".js-contact").forEach(function (a) {
    if (accessMail) {
      a.href = accessMail;
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
  function menuOpen() { return toggle.getAttribute("aria-expanded") === "true"; }
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape" || !menuOpen()) return;
    // Hand focus back to the toggle, or it would be lost on a now-hidden link.
    var inside = links.contains(document.activeElement);
    setMenu(false);
    if (inside || document.activeElement === toggle) toggle.focus();
  });
  // Close when keyboard focus or a tap moves outside the menu.
  document.addEventListener("focusin", function (e) {
    if (menuOpen() && !links.contains(e.target) && e.target !== toggle) setMenu(false);
  });
  document.addEventListener("click", function (e) {
    if (menuOpen() && !links.contains(e.target) && !toggle.contains(e.target)) setMenu(false);
  });
  // Reset the menu when the window grows to the desktop layout.
  if (window.matchMedia) {
    var desktop = matchMedia("(min-width: 821px)");
    var onDesktop = function () { if (desktop.matches) setMenu(false); };
    if (desktop.addEventListener) desktop.addEventListener("change", onDesktop);
    else if (desktop.addListener) desktop.addListener(onDesktop);
  }

  // ---------- Motion ----------
  var root = document.documentElement;
  var motion = root.classList.contains("motion");

  // Scroll reveal: a short, quiet fade, staggered inside each group.
  var revealEls = Array.prototype.slice.call(document.querySelectorAll("[data-reveal]"));
  function reveal(el) { el.classList.add("is-in"); }
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
