"use strict";

// The theme switch: System (follow the device), Light or Dark.
// This file is loaded in the head of the page, so a saved choice is applied before the first paint and the page
// never flashes the wrong theme. The choice is the only thing kept in the browser's local storage.

const THEME_KEY = "chat-with-any-website-agent:theme";
const THEME_CHOICES = ["system", "light", "dark"];

function normalizeChoice(value) {
  return THEME_CHOICES.includes(value) ? value : "system";
}

// `storage` may be missing or may throw (private browsing, blocked site data): the page then simply follows the device.
function loadChoice(storage) {
  try {
    return normalizeChoice(storage && storage.getItem(THEME_KEY));
  } catch {
    return "system";
  }
}

function saveChoice(storage, choice) {
  try {
    if (choice === "system") storage.removeItem(THEME_KEY);
    else storage.setItem(THEME_KEY, choice);
  } catch {
    // The choice still applies until the page is closed.
  }
}

// "system" removes data-theme, so the stylesheet falls back to the setting of the device.
function applyChoice(root, choice) {
  if (choice === "system") delete root.dataset.theme;
  else root.dataset.theme = choice;
}

function markChoice(buttons, choice) {
  for (const button of buttons) button.setAttribute("aria-pressed", String(button.dataset.choice === choice));
}

// Shows the switch (it is hidden until this script runs) and makes its buttons change and remember the theme.
function wireSwitch(container, buttons, root, storage) {
  markChoice(buttons, loadChoice(storage));
  for (const button of buttons) {
    button.addEventListener("click", () => {
      const choice = normalizeChoice(button.dataset.choice);
      applyChoice(root, choice);
      saveChoice(storage, choice);
      markChoice(buttons, choice);
    });
  }
  container.hidden = false;
}

function browserStorage() {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

if (typeof document !== "undefined" && document.documentElement && typeof window !== "undefined") {
  applyChoice(document.documentElement, loadChoice(browserStorage()));
  document.addEventListener("DOMContentLoaded", () => {
    const container = document.getElementById("theme-switch");
    if (container) {
      wireSwitch(container, [...container.querySelectorAll("button[data-choice]")], document.documentElement, browserStorage());
    }
  });
}
