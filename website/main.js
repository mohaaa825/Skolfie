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
