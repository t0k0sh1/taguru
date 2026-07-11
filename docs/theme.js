// Theme bootstrap + toggle. Loaded synchronously in <head> so a saved
// choice applies before first paint (no flash), on file:// and Pages
// alike. With no saved choice the CSS prefers-color-scheme rules run.
(function () {
  var saved = null;
  try { saved = localStorage.getItem("taguru-docs-theme"); } catch (e) {}
  if (saved === "light" || saved === "dark") {
    document.documentElement.setAttribute("data-theme", saved);
  }
  window.addEventListener("DOMContentLoaded", function () {
    var button = document.getElementById("theme-toggle");
    if (!button) return;
    button.addEventListener("click", function () {
      var root = document.documentElement;
      var current = root.getAttribute("data-theme");
      if (current !== "light" && current !== "dark") {
        current = window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark" : "light";
      }
      var next = current === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("taguru-docs-theme", next); } catch (e) {}
    });
  });
})();
