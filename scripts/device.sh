# Sourced by the launch scripts. Puts the device in the conditions the paper's
# latencies were measured under, then runs the workload.
#
#   Raspberry Pi: performance governor, frequency locked at its maximum, page
#                 cache dropped before every run. Restored on exit.
#   macOS:        highest scheduling priority (nice -n -20).
#   Other hosts:  no tuning.
#
# The tuning needs root, so launch the scripts with sudo -E (-E keeps WANDB_* and
# the rest of your environment). The workload itself runs as the user who called
# sudo, with the priority inherited: outputs, the uv cache and the W&B login stay
# theirs instead of root's.
#
#   DEVICE_TUNING=0  skip the tuning (the latencies are then not comparable).
#   RPI_RT_CPU=<n>   Raspberry Pi only, opt-in: also pin to core <n> at real-time
#                    priority (chrt -f 99). Off by default, as in the reference
#                    script: ONNX Runtime runs 8 threads, so on one core they
#                    compete, and a FIFO-99 CPU-bound process for hours can
#                    starve the rest of the system (SSH included).

DEVICE_KIND=other
if [[ "$(uname -s)" == "Darwin" ]]; then
    DEVICE_KIND=mac
elif [[ -r /proc/device-tree/model ]] && tr -d '\0' </proc/device-tree/model | grep -qi "raspberry pi"; then
    DEVICE_KIND=rpi
fi

DEVICE_USER="${SUDO_USER:-$(id -un)}"
_GOVERNORS=()
_MIN_FREQS=()

_restore_rpi() {
    local i=0 cpu
    for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do
        [[ -n "${_GOVERNORS[$i]:-}" ]] && echo "${_GOVERNORS[$i]}" >"$cpu/scaling_governor"
        [[ -n "${_MIN_FREQS[$i]:-}" ]] && echo "${_MIN_FREQS[$i]}" >"$cpu/scaling_min_freq"
        i=$((i + 1))
    done
    echo "[+] CPU governor and minimum frequency restored"
}

device_setup() {
    if [[ "${DEVICE_TUNING:-1}" == "0" || "$DEVICE_KIND" == "other" ]]; then
        echo "[+] ${DEVICE_KIND}: no device tuning"
        return
    fi
    if [[ $EUID -ne 0 ]]; then
        echo "Device tuning needs root: run with 'sudo -E $0 $*'" >&2
        echo "(or DEVICE_TUNING=0 to skip it; latencies will not be comparable)" >&2
        exit 1
    fi

    if [[ "$DEVICE_KIND" == "rpi" ]]; then
        local cpu max_freq
        for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do
            _GOVERNORS+=("$(cat "$cpu/scaling_governor")")
            _MIN_FREQS+=("$(cat "$cpu/scaling_min_freq")")
        done
        trap _restore_rpi EXIT

        echo "[+] Forcing CPU to maximum performance..."
        echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
        # Also lock the frequency: even in performance mode it can oscillate.
        max_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
        echo "[+] Locking frequency at ${max_freq} kHz..."
        echo "$max_freq" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq >/dev/null
    fi
}

# Absolute path to the user's uv: sudo resets PATH and would not find it.
device_uv() {
    if [[ $EUID -eq 0 && -n "${SUDO_USER:-}" ]]; then
        sudo -u "$DEVICE_USER" -H bash -lc 'command -v uv'
    else
        command -v uv
    fi
}

# Run a command under the device's tuning, as the invoking user.
device_exec() {
    local prefix=()
    if [[ "${DEVICE_TUNING:-1}" != "0" && $EUID -eq 0 ]]; then
        case "$DEVICE_KIND" in
            rpi)
                sync
                echo 3 >/proc/sys/vm/drop_caches
                if [[ -n "${RPI_RT_CPU:-}" ]]; then
                    echo "[+] Core ${RPI_RT_CPU}, real-time priority (FIFO 99)"
                    prefix=(chrt -f 99 taskset -c "$RPI_RT_CPU")
                fi
                ;;
            mac)
                prefix=(nice -n -20)
                ;;
        esac
    fi
    if [[ $EUID -eq 0 && -n "${SUDO_USER:-}" ]]; then
        # Scheduling priority and CPU affinity are inherited through sudo -u.
        ${prefix[@]+"${prefix[@]}"} sudo -u "$DEVICE_USER" -H \
            --preserve-env=WANDB_MODE,WANDB_API_KEY,WANDB_ENTITY,WANDB_PROJECT,AWS_PROFILE,AWS_SHARED_CREDENTIALS_FILE \
            "$@"
    else
        ${prefix[@]+"${prefix[@]}"} "$@"
    fi
}
