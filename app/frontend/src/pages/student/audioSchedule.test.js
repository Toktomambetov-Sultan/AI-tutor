import test from 'node:test';
import assert from 'node:assert/strict';

import { getScheduledStartAt, getSubtitleDelayMs } from './audioSchedule.js';

test('getScheduledStartAt prefers immediate preroll when queue is stale', () => {
    const startAt = getScheduledStartAt(10, 9.8, 0.05, 0.02);
    assert.equal(startAt, 10.02);
});

test('getScheduledStartAt keeps overlap when queue is ahead', () => {
    const startAt = getScheduledStartAt(10, 10.5, 0.05, 0.02);
    assert.equal(startAt, 10.45);
});

test('getSubtitleDelayMs clamps negative delay to zero', () => {
    const delayMs = getSubtitleDelayMs(4.99, 5);
    assert.equal(delayMs, 0);
});

test('getSubtitleDelayMs converts seconds to rounded milliseconds', () => {
    const delayMs = getSubtitleDelayMs(5.1236, 5);
    assert.equal(delayMs, 124);
});
