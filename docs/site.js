// Site chrome: theme bootstrap + toggle, and the mobile sidebar.
// Loaded synchronously in <head> so a saved theme applies before first
// paint (no flash), on file:// and Pages alike. With no saved choice
// the CSS prefers-color-scheme rules run.
(function () {
  var saved = null;
  try { saved = localStorage.getItem("taguru-docs-theme"); } catch (e) {}
  if (saved === "light" || saved === "dark") {
    document.documentElement.setAttribute("data-theme", saved);
  }
  window.addEventListener("DOMContentLoaded", function () {
    function currentTheme() {
      var t = document.documentElement.getAttribute("data-theme");
      if (t !== "light" && t !== "dark") {
        t = window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark" : "light";
      }
      return t;
    }
    Array.prototype.forEach.call(
      document.querySelectorAll(".theme-toggle"),
      function (button) {
        button.addEventListener("click", function () {
          var next = currentTheme() === "dark" ? "light" : "dark";
          document.documentElement.setAttribute("data-theme", next);
          try { localStorage.setItem("taguru-docs-theme", next); } catch (e) {}
        });
      });
    var navToggle = document.getElementById("nav-toggle");
    var scrim = document.getElementById("scrim");
    if (navToggle) {
      navToggle.addEventListener("click", function () {
        document.body.classList.toggle("nav-open");
      });
    }
    if (scrim) {
      scrim.addEventListener("click", function () {
        document.body.classList.remove("nav-open");
      });
    }
  });
})();
