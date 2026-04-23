export function getScheduledStartAt(currentTime, nextStart, crossfadeSec, prerollSec) {
    return Math.max(currentTime + prerollSec, nextStart - crossfadeSec);
}

export function getSubtitleDelayMs(startAt, currentTime) {
    return Math.max(0, Math.round((startAt - currentTime) * 1000));
}
