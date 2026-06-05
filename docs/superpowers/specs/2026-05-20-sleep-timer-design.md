# Sleep Timer — Design Spec
_2026-05-20_

## Overview

Add a sleep timer to the book player (`listen_book.html`) that pauses playback after a user-chosen duration. All changes are frontend-only; no backend or Django modifications required.

## UI

### Header button

A moon icon button (`🌙`) is added to the right side of the header, between the existing "Script" and "← Library" links.

- **Inactive:** subdued text/icon, no background
- **Active:** dark pill (`background: #1a1a1a; color: #fff`) showing a live countdown (`🌙 28:14`), updated every second

Clicking the button (either state) opens the popover.

### Popover

A small card anchored below the header button with an upward-pointing caret. Dismissed by clicking outside or selecting a duration.

Contents:
1. **Label row:** "Sleep timer" in small-caps; when active, shows remaining time (`· 28:14 left`) inline
2. **Duration grid:** 3×2 square pill buttons — `15m`, `30m`, `45m`, `60m`, `2h`, `Off`
   - The active duration button is highlighted (dark fill)
   - Selecting any duration starts/restarts the timer and closes the popover
   - `Off` cancels the timer and closes the popover
3. **Divider**
4. **Checkbox:** "Finish current chapter first" — when checked, the timer waits for the current audio track to end before pausing rather than pausing immediately on expiry

## Behaviour

### Starting / changing the timer

Clicking a duration pill:
- Clears any existing `setTimeout` / `setInterval`
- Starts a new `setTimeout` for the chosen duration
- Starts a `setInterval` (1 s tick) that updates the header button countdown
- Closes the popover

### On expiry

- **Checkbox unchecked:** call `player.pause()` immediately; clear the countdown interval; reset button to inactive state
- **Checkbox checked:** set a boolean flag `stopAfterChapter = true`; the existing `ended` event handler is extended to check this flag — if set, it skips auto-advancing to the next chapter, calls `player.pause()`, clears the flag, and resets the button

### Cancelling

Clicking `Off` in the popover, or the timer completing, both result in the same teardown: clear `setTimeout` + `setInterval`, reset button to inactive, clear `stopAfterChapter` flag.

### Page refresh

The active timer is not persisted — refresh resets to no timer. This is standard sleep-timer behaviour.

## Persistence

`localStorage` key `reader_sleep_finish_chapter` stores the checkbox boolean. Loaded on page init; written on every checkbox change.

## Implementation scope

All changes confined to `reader/templates/reader/listen_book.html`:

- ~10 lines of CSS (button inactive/active states, popover card, caret, duration grid, checkbox row)
- ~10 lines of HTML (header button, popover markup)
- ~60 lines of JS (timer state, open/close popover, countdown tick, expiry handler, integration with existing `ended` handler)

No new files. No Django views, models, or URL changes.
