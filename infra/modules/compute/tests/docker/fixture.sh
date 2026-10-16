#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    printf '%s\n' 'Expected one fixture mode: service, smoke-pass, smoke-fail, or hold.' >&2
    exit 64
fi

require_smoke_contract() {
    if [ "${ALIGN_ENVIRONMENT-}" != dev ] ||
       [ "${ALIGN_SMOKE_OFFLINE-}" != true ] ||
       [ -z "${ALIGN_RUN_ID-}" ] ||
       [ "${ALIGN_OUTPUT_PREFIX-}" != "smoke/${ALIGN_RUN_ID-}/" ]; then
        printf '%s\n' 'Invalid fixture smoke environment.' >&2
        exit 64
    fi
}

trap 'exit 0' TERM INT

case "$1" in
    service)
        printf '%s\n' 'ALIGN fixture service ready'
        while :; do sleep 1; done
        ;;
    smoke-pass)
        require_smoke_contract
        printf '%s\n' 'ALIGN smoke fixture passed'
        exit 0
        ;;
    smoke-fail)
        require_smoke_contract
        printf '%s\n' 'ALIGN smoke fixture failed intentionally' >&2
        exit 7
        ;;
    hold)
        require_smoke_contract
        printf '%s\n' 'ALIGN fixture hold ready'
        while [ ! -f /tmp/align-release ]; do sleep 1; done
        ;;
    *)
        printf '%s\n' 'Unknown fixture mode.' >&2
        exit 64
        ;;
esac
