#!/usr/bin/env bash
set -euo pipefail

readonly sdk_header_sha256="0beeec96f323c0bc87df707940f379f3dca0ec5b31bea9e49b90979ad7bb5b62"
readonly sdk_types_sha256="f20bfe782e043c8aaa2f7e89eee6df6b701ecb7711150f7ab3d7a1a66d86d892"
readonly sdk_initializers_sha256="4e9c6607138ece3351eaf072634920ba1b16bde89483151c50700713745062e9"
readonly sdk_remote_sha256="91b42c423e36031c7bade01964c341c9f9423b28aeadd300056792d982c0ccaf"
readonly sdk_integrated_sha256="0e67141b97b64c089c3bbdab47980ca9822c4de19adea810a2f68722adcb3fe3"
readonly bundle_marker_name=".dex-teleop-manus-bridge-bundle"
readonly bundle_kind="dex_teleop_manus_bridge_bundle"
readonly bundle_schema="1"

usage() {
    cat <<'EOF'
Usage: build_manus_bridge.sh [OPTIONS]

Build and transactionally publish one explicit MANUS SDK sidecar variant
without copying proprietary MANUS files into this repository.

Options:
  --mode integrated|remote
  --sdk-root PATH
  --output PATH
  --runtime-prefix PATH
  --allow-owned-runtime-prefix-outside-cache

Integrated is the backwards-compatible default. Remote requires the strictly
validated runtime produced by build_manus_remote_runtime.sh. MANUS_SDK_ROOT,
DEX_TELEOP_MANUS_BRIDGE_OUTPUT, and DEX_TELEOP_MANUS_REMOTE_RUNTIME provide
equivalent defaults. The Remote runtime prefix is ignored by Integrated.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
sdk_root="${MANUS_SDK_ROOT:-}"
mode="integrated"
output="${DEX_TELEOP_MANUS_BRIDGE_OUTPUT:-}"
remote_runtime_prefix="${DEX_TELEOP_MANUS_REMOTE_RUNTIME:-${XDG_CACHE_HOME:-${HOME}/.cache}/dex_teleop/manus_sdk_v3.1.1_remote_runtime}"
allow_owned_runtime_prefix_outside_cache=false

while (($#)); do
    case "$1" in
        --mode)
            [[ $# -ge 2 ]] || { echo "--mode requires integrated or remote" >&2; exit 2; }
            mode="$2"
            shift 2
            ;;
        --sdk-root)
            [[ $# -ge 2 ]] || { echo "--sdk-root requires a path" >&2; exit 2; }
            sdk_root="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || { echo "--output requires a path" >&2; exit 2; }
            output="$2"
            shift 2
            ;;
        --runtime-prefix)
            [[ $# -ge 2 ]] || { echo "--runtime-prefix requires a path" >&2; exit 2; }
            remote_runtime_prefix="$2"
            shift 2
            ;;
        --allow-owned-runtime-prefix-outside-cache)
            allow_owned_runtime_prefix_outside_cache=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$mode" in
    integrated|remote) ;;
    *)
        echo "--mode requires integrated or remote, got: $mode" >&2
        exit 2
        ;;
esac

for tool in awk cp find g++ ldd mktemp readelf readlink realpath sha256sum wc; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required bridge build tool is unavailable: $tool" >&2
        exit 3
    fi
done

if [[ -z "$output" ]]; then
    output="${XDG_CACHE_HOME:-${HOME}/.cache}/dex_teleop/manus_bridge/manus_bridge"
    if [[ "$mode" == "remote" ]]; then
        output="${output}_remote"
    fi
fi
if [[ "$output" == *$'\n'* || "$sdk_root" == *$'\n'* ||
      "$remote_runtime_prefix" == *$'\n'* ]]; then
    echo "Paths containing newlines are not supported." >&2
    exit 2
fi

test_failure_phase="${DEX_TELEOP_MANUS_BRIDGE_TEST_FAIL_PHASE:-}"
if [[ -n "$test_failure_phase" ]]; then
    if [[ "${DEX_TELEOP_MANUS_BRIDGE_TESTING:-}" != "1" ]]; then
        echo "Publication failure injection requires DEX_TELEOP_MANUS_BRIDGE_TESTING=1." >&2
        exit 2
    fi
    case "$test_failure_phase" in
        before-runtime-publish|after-runtime-publish|after-executable-publish|final-verification) ;;
        *)
            echo "Unknown publication failure-injection phase: $test_failure_phase" >&2
            exit 2
            ;;
    esac
fi

file_sha256() {
    sha256sum -- "$1" | awk '{ print $1 }'
}

metadata_value() {
    local key="$1"
    local file="$2"
    awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$file" 2>/dev/null || true
}

sdk_layout_is_v3_1_1() {
    local candidate="$1"
    [[ -f "$candidate/include/ManusSDK.h" ]] || return 1
    [[ -f "$candidate/include/ManusSDKTypes.h" ]] || return 1
    [[ -f "$candidate/include/ManusSDKTypeInitializers.h" ]] || return 1
    [[ -f "$candidate/lib/libManusSDK.so" ]] || return 1
    [[ -f "$candidate/lib/libManusSDK_Integrated.so" ]] || return 1
    [[ "$(file_sha256 "$candidate/include/ManusSDK.h")" == "$sdk_header_sha256" ]] || return 1
    [[ "$(file_sha256 "$candidate/include/ManusSDKTypes.h")" == "$sdk_types_sha256" ]] || return 1
    [[ "$(file_sha256 "$candidate/include/ManusSDKTypeInitializers.h")" == "$sdk_initializers_sha256" ]] || return 1
    [[ "$(file_sha256 "$candidate/lib/libManusSDK.so")" == "$sdk_remote_sha256" ]] || return 1
    [[ "$(file_sha256 "$candidate/lib/libManusSDK_Integrated.so")" == "$sdk_integrated_sha256" ]] || return 1
}

candidate_roots=()
if [[ -n "$sdk_root" ]]; then
    candidate_roots+=("$(realpath -m -- "$sdk_root")")
fi
candidate_roots+=(
    "$(realpath -m -- "${HOME}/manus_setup")"
    "$(realpath -m -- "${HOME}/Desktop/emg/manus")"
)

include_dir=""
sdk_lib=""
sdk_layout=""
searched=()
for root in "${candidate_roots[@]}"; do
    [[ -d "$root" ]] || continue
    candidates=(
        "$root"
        "$root/ManusSDK"
        "$root/SDKClient_Linux/ManusSDK"
        "$root/SDKMinimalClient_Linux/ManusSDK"
    )
    while IFS= read -r found; do
        candidates+=("$found")
    done < <(find "$root/vendor" -maxdepth 4 -type d -name ManusSDK 2>/dev/null | sort)

    for candidate in "${candidates[@]}"; do
        candidate="$(realpath -m -- "$candidate")"
        searched+=("$candidate")
        if sdk_layout_is_v3_1_1 "$candidate"; then
            sdk_layout="$candidate"
            include_dir="$candidate/include"
            if [[ "$mode" == "integrated" ]]; then
                sdk_lib="$candidate/lib/libManusSDK_Integrated.so"
            else
                sdk_lib="$candidate/lib/libManusSDK.so"
            fi
            break 2
        fi
    done
done

if [[ -z "$sdk_layout" ]]; then
    echo "Could not find a fingerprint-verified MANUS SDK v3.1.1 layout." >&2
    echo "Set MANUS_SDK_ROOT or pass --sdk-root. Searched:" >&2
    printf '  %s\n' "${searched[@]}" >&2
    exit 3
fi

remote_runtime_lib=""
if [[ "$mode" == "remote" ]]; then
    runtime_command=(
        bash
        "$script_dir/build_manus_remote_runtime.sh"
        --verify-only
        --prefix
        "$remote_runtime_prefix"
    )
    if [[ "$allow_owned_runtime_prefix_outside_cache" == true ]]; then
        runtime_command+=(--allow-owned-prefix-outside-cache)
    fi
    if ! runtime_verification="$("${runtime_command[@]}" 2>&1)"; then
        echo "MANUS Remote runtime failed strict validation:" >&2
        printf '%s\n' "$runtime_verification" >&2
        exit 4
    fi
    remote_runtime_prefix="$(realpath -e -- "$remote_runtime_prefix")"
    remote_runtime_lib="$remote_runtime_prefix/lib"
fi

if [[ "$mode" == "remote" ]]; then
    vendor_ldd="$(env -u LD_PRELOAD LD_LIBRARY_PATH="$remote_runtime_lib" \
        ldd -r "$sdk_lib" 2>&1 || true)"
else
    vendor_ldd="$(env -u LD_LIBRARY_PATH -u LD_PRELOAD ldd -r "$sdk_lib" 2>&1 || true)"
fi
closure_pattern='not found|undefined symbol'
if [[ "$vendor_ldd" =~ $closure_pattern ]]; then
    echo "MANUS $mode SDK v3.1.1 failed load/relocation validation:" >&2
    printf '%s\n' "$vendor_ldd" >&2
    exit 4
fi

output_parent="$(dirname -- "$output")"
output_name="$(basename -- "$output")"
mkdir -p -- "$output_parent"
output_parent="$(cd -- "$output_parent" && pwd -P)"
output="$output_parent/$output_name"
if [[ -d "$output" && ! -L "$output" ]]; then
    echo "Bridge output is an existing directory, refusing replacement: $output" >&2
    exit 4
fi

bundle_root="$output_parent/.dex_teleop_manus_bridge_bundles"
mkdir -p -- "$bundle_root"
staging_bundle="$(mktemp -d "$bundle_root/.${output_name}.${mode}.stage.XXXXXX")"
staged_output="$staging_bundle/$output_name"
staged_runtime_dir="$staging_bundle/manus_runtime_${mode}"
mkdir -p -- "$staged_runtime_dir"
ln -s -- "$sdk_lib" "$staged_runtime_dir/libManusSDK.so"

cat > "$staging_bundle/$bundle_marker_name" <<EOF
kind=$bundle_kind
schema=$bundle_schema
mode=$mode
sdk_library_sha256=$(file_sha256 "$sdk_lib")
EOF

bundle_to_cleanup="$staging_bundle"
runtime_path="$output_parent/manus_runtime_${mode}"
runtime_publish_link=""
publish_executable=""
previous_executable_backup=""
legacy_runtime_backup=""
previous_executable_present=false
previous_executable_sha256=""
previous_runtime_kind="absent"
previous_runtime_link_value=""
previous_runtime_target=""
old_bundle=""
published_bundle=""
artifact_sha256=""
state_snapshotted=false
runtime_published=false
executable_published=false
publish_committed=false
legacy_runtime_moved=false
expected_sdk_library_sha256="$(file_sha256 "$sdk_lib")"

bundle_is_owned() {
    local bundle="$1"
    local marker="$bundle/$bundle_marker_name"
    [[ -f "$marker" ]] || return 1
    [[ "$(metadata_value kind "$marker")" == "$bundle_kind" ]] || return 1
    [[ "$(metadata_value schema "$marker")" == "$bundle_schema" ]] || return 1
    [[ "$(metadata_value mode "$marker")" == "$mode" ]] || return 1
    [[ "$(metadata_value sdk_library_sha256 "$marker")" == \
       "$expected_sdk_library_sha256" ]] || return 1
    [[ -L "$bundle/manus_runtime_${mode}/libManusSDK.so" ]] || return 1
    [[ "$(readlink -f -- "$bundle/manus_runtime_${mode}/libManusSDK.so")" == \
       "$sdk_lib" ]] || return 1
}

legacy_runtime_is_owned() {
    local runtime="$1"
    [[ -d "$runtime" && ! -L "$runtime" ]] || return 1
    [[ -L "$runtime/libManusSDK.so" ]] || return 1
    [[ "$(readlink -f -- "$runtime/libManusSDK.so")" == "$sdk_lib" ]] || return 1
    local entry_count
    entry_count="$(find "$runtime" -mindepth 1 -maxdepth 1 -printf '.' | wc -c)"
    [[ "$entry_count" == "1" ]]
}

maybe_fail_publication() {
    local phase="$1"
    if [[ "$test_failure_phase" == "$phase" ]]; then
        echo "Injected MANUS bridge publication failure at phase: $phase" >&2
        exit 5
    fi
}

rollback_publication() {
    local rollback_ok=true
    local rollback_link=""

    if [[ "$legacy_runtime_moved" == true && "$runtime_published" != true &&
          "$previous_runtime_kind" == "legacy_directory" ]]; then
        if [[ ! -e "$runtime_path" && -d "$legacy_runtime_backup" ]] &&
           mv -- "$legacy_runtime_backup" "$runtime_path"; then
            legacy_runtime_backup=""
            legacy_runtime_moved=false
        else
            echo "Failed to restore legacy MANUS runtime before publication." >&2
            rollback_ok=false
        fi
    fi

    if [[ "$executable_published" == true ]]; then
        if [[ "$previous_executable_present" == true ]]; then
            if [[ ! -f "$previous_executable_backup" ||
                  "$(file_sha256 "$previous_executable_backup")" != \
                      "$previous_executable_sha256" ]]; then
                echo "Cannot roll back MANUS bridge: executable snapshot is invalid." >&2
                rollback_ok=false
            elif ! mv -Tf -- "$previous_executable_backup" "$output"; then
                echo "Cannot roll back MANUS bridge executable: $output" >&2
                rollback_ok=false
            else
                previous_executable_backup=""
                executable_published=false
            fi
        elif [[ -f "$output" && ! -L "$output" &&
                "$(file_sha256 "$output")" == "$artifact_sha256" ]]; then
            if rm -f -- "$output"; then
                executable_published=false
            else
                echo "Cannot remove newly published MANUS bridge: $output" >&2
                rollback_ok=false
            fi
        else
            echo "Refusing to roll back an unexpected MANUS bridge executable state." >&2
            rollback_ok=false
        fi
    fi

    if [[ "$runtime_published" == true ]]; then
        if [[ ! -L "$runtime_path" ||
              "$(readlink -f -- "$runtime_path" 2>/dev/null || true)" != \
                  "$published_bundle/manus_runtime_${mode}" ]] ||
           ! bundle_is_owned "$published_bundle"; then
            echo "Refusing to replace an unexpected or unowned runtime link during rollback." >&2
            rollback_ok=false
        else
            case "$previous_runtime_kind" in
                owned_symlink)
                    if [[ -z "$old_bundle" ]] || ! bundle_is_owned "$old_bundle"; then
                        echo "Cannot restore previous unowned MANUS runtime bundle." >&2
                        rollback_ok=false
                    else
                        rollback_link="$output_parent/.manus_runtime_${mode}.rollback.$$.$RANDOM"
                        if ln -s -- "$previous_runtime_link_value" "$rollback_link" &&
                           [[ "$(readlink -f -- "$rollback_link" 2>/dev/null || true)" == \
                              "$previous_runtime_target" ]] &&
                           mv -Tf -- "$rollback_link" "$runtime_path"; then
                            rollback_link=""
                            runtime_published=false
                        else
                            echo "Failed to restore the exact previous MANUS runtime link." >&2
                            rollback_ok=false
                        fi
                    fi
                    ;;
                legacy_directory)
                    if rm -f -- "$runtime_path" &&
                       [[ -d "$legacy_runtime_backup" ]] &&
                       mv -- "$legacy_runtime_backup" "$runtime_path"; then
                        legacy_runtime_backup=""
                        legacy_runtime_moved=false
                        runtime_published=false
                    else
                        echo "Failed to restore the previous legacy MANUS runtime directory." >&2
                        rollback_ok=false
                    fi
                    ;;
                absent)
                    if rm -f -- "$runtime_path"; then
                        runtime_published=false
                    else
                        echo "Failed to remove newly published MANUS runtime link." >&2
                        rollback_ok=false
                    fi
                    ;;
                *)
                    echo "Unknown previous MANUS runtime state: $previous_runtime_kind" >&2
                    rollback_ok=false
                    ;;
            esac
        fi
    fi
    if [[ -n "$rollback_link" && ( -e "$rollback_link" || -L "$rollback_link" ) ]]; then
        rm -f -- "$rollback_link"
    fi

    if [[ "$previous_executable_present" == true ]]; then
        if [[ ! -f "$output" || -L "$output" ||
              "$(file_sha256 "$output")" != "$previous_executable_sha256" ]]; then
            echo "MANUS bridge executable rollback verification failed." >&2
            rollback_ok=false
        fi
        case "$previous_runtime_kind" in
            owned_symlink)
                if [[ ! -L "$runtime_path" ||
                      "$(readlink -- "$runtime_path")" != \
                          "$previous_runtime_link_value" ||
                      "$(readlink -f -- "$runtime_path")" != \
                          "$previous_runtime_target" ]]; then
                    echo "MANUS runtime-link rollback verification failed." >&2
                    rollback_ok=false
                fi
                ;;
            legacy_directory)
                if ! legacy_runtime_is_owned "$runtime_path"; then
                    echo "Legacy MANUS runtime rollback verification failed." >&2
                    rollback_ok=false
                fi
                ;;
        esac
        if [[ "$rollback_ok" == true ]] && ! verify_bridge "$output"; then
            echo "Restored MANUS bridge pair failed loader/protocol verification." >&2
            rollback_ok=false
        fi
    elif [[ -e "$output" || -L "$output" || -e "$runtime_path" ||
            -L "$runtime_path" ]]; then
        echo "Rollback did not restore the previous absent publication state." >&2
        rollback_ok=false
    fi

    [[ "$rollback_ok" == true ]]
}

cleanup() {
    local status=$?
    trap - EXIT
    set +e
    local rollback_ok=true
    if [[ -n "$publish_executable" &&
          ( -e "$publish_executable" || -L "$publish_executable" ) ]]; then
        rm -f -- "$publish_executable"
    fi
    if [[ -n "$runtime_publish_link" &&
          ( -e "$runtime_publish_link" || -L "$runtime_publish_link" ) ]]; then
        rm -f -- "$runtime_publish_link"
    fi
    if [[ "$publish_committed" != true && "$state_snapshotted" == true ]]; then
        rollback_publication || rollback_ok=false
    fi
    if [[ -n "$previous_executable_backup" &&
          -f "$previous_executable_backup" ]]; then
        rm -f -- "$previous_executable_backup"
    fi
    if [[ "$publish_committed" != true && "$rollback_ok" == true &&
          -n "$bundle_to_cleanup" &&
          -d "$bundle_to_cleanup" ]] &&
       bundle_is_owned "$bundle_to_cleanup"; then
        rm -rf -- "$bundle_to_cleanup"
    fi
    if [[ "$rollback_ok" != true ]]; then
        echo "MANUS bridge publication rollback failed; preserved owned artifacts for recovery." >&2
        exit 6
    fi
    exit "$status"
}
trap cleanup EXIT

mode_define="DEX_TELEOP_MANUS_LINK_MODE_INTEGRATED=1"
if [[ "$mode" == "remote" ]]; then
    mode_define="DEX_TELEOP_MANUS_LINK_MODE_REMOTE=1"
fi

cxx="${CXX:-g++}"
runtime_rpath="\$ORIGIN/manus_runtime_${mode}"
remote_linker_args=()
if [[ "$mode" == "remote" ]]; then
    runtime_rpath="$runtime_rpath:$remote_runtime_lib"
    remote_linker_args+=("-Wl,-rpath-link,$remote_runtime_lib")
fi
"$cxx" \
    -std=c++17 \
    -O2 \
    -Wall \
    -Wextra \
    -Wpedantic \
    -pthread \
    -D"$mode_define" \
    -I"$include_dir" \
    "$script_dir/manus_bridge.cpp" \
    -L"$staged_runtime_dir" \
    -Wl,--disable-new-dtags \
    -Wl,-rpath,"$runtime_rpath" \
    "${remote_linker_args[@]}" \
    -Wl,-z,defs \
    -lManusSDK \
    -o "$staged_output"

verify_bridge() {
    local bridge="$1"
    local dynamic loader probe
    dynamic="$(readelf -d "$bridge")"
    [[ "$dynamic" == *"(RPATH)"* ]] || return 1
    [[ "$dynamic" != *"(RUNPATH)"* ]] || return 1
    [[ "$dynamic" == *"Library rpath: [$runtime_rpath]"* ]] || return 1
    loader="$(env -u LD_LIBRARY_PATH -u LD_PRELOAD ldd -r "$bridge" 2>&1 || true)"
    if [[ "$loader" =~ $closure_pattern ]]; then
        printf '%s\n' "$loader" >&2
        return 1
    fi
    probe="$(env -u LD_LIBRARY_PATH -u LD_PRELOAD "$bridge" --protocol-version)"
    [[ "$probe" == "dex_teleop.manus 2" ]] || return 1
}

if ! verify_bridge "$staged_output"; then
    echo "Staged MANUS $mode bridge failed RPATH, relocation, or protocol verification." >&2
    exit 5
fi

artifact_sha256="$(file_sha256 "$staged_output")"
printf 'executable_sha256=%s\n' "$artifact_sha256" >> \
    "$staging_bundle/$bundle_marker_name"
bundle_name="${output_name}.${mode}.${artifact_sha256:0:16}.$$"
published_bundle="$bundle_root/$bundle_name"
if [[ -e "$published_bundle" || -L "$published_bundle" ]]; then
    echo "Refusing to overwrite an existing bridge bundle: $published_bundle" >&2
    exit 5
fi
mv -- "$staging_bundle" "$published_bundle"
bundle_to_cleanup="$published_bundle"
staged_output="$published_bundle/$output_name"

if [[ -e "$output" || -L "$output" ]]; then
    if [[ ! -f "$output" || -L "$output" ]]; then
        echo "Existing MANUS bridge executable is not an owned regular artifact: $output" >&2
        exit 5
    fi
    previous_executable_present=true
    if [[ -L "$runtime_path" ]]; then
        previous_runtime_kind="owned_symlink"
        previous_runtime_link_value="$(readlink -- "$runtime_path")"
        previous_runtime_target="$(readlink -f -- "$runtime_path" 2>/dev/null || true)"
        if [[ "$previous_runtime_target" != \
              "$bundle_root/"*"/manus_runtime_${mode}" ]]; then
            echo "Existing runtime link is outside the managed bridge bundles: $runtime_path" >&2
            exit 5
        fi
        old_bundle="$(dirname -- "$previous_runtime_target")"
        if ! bundle_is_owned "$old_bundle"; then
            echo "Existing runtime link points to an unowned bundle: $runtime_path" >&2
            exit 5
        fi
    elif [[ -e "$runtime_path" ]]; then
        previous_runtime_kind="legacy_directory"
        if ! legacy_runtime_is_owned "$runtime_path"; then
            echo "Existing runtime directory is not owned by this bridge builder: $runtime_path" >&2
            exit 5
        fi
    else
        echo "Existing MANUS bridge has no matching owned runtime: $output" >&2
        exit 5
    fi

    if ! verify_bridge "$output"; then
        echo "Existing MANUS bridge/runtime pair is not loader/protocol viable." >&2
        exit 5
    fi
    previous_executable_sha256="$(file_sha256 "$output")"
    if [[ -n "$old_bundle" ]]; then
        recorded_executable_sha256="$(
            metadata_value executable_sha256 "$old_bundle/$bundle_marker_name"
        )"
        if [[ -n "$recorded_executable_sha256" &&
              "$recorded_executable_sha256" != "$previous_executable_sha256" ]]; then
            echo "Existing MANUS executable does not match its owned bundle marker." >&2
            exit 5
        fi
    fi
    previous_executable_backup="$output_parent/.${output_name}.previous.$$.$RANDOM"
    if [[ -e "$previous_executable_backup" || -L "$previous_executable_backup" ]]; then
        echo "Refusing to overwrite an existing executable snapshot: $previous_executable_backup" >&2
        exit 5
    fi
    cp --preserve=mode,timestamps -- "$output" "$previous_executable_backup"
    if [[ "$(file_sha256 "$previous_executable_backup")" != \
          "$previous_executable_sha256" ]]; then
        echo "MANUS bridge executable snapshot verification failed." >&2
        exit 5
    fi
elif [[ -e "$runtime_path" || -L "$runtime_path" ]]; then
    echo "Refusing inconsistent MANUS publication state: runtime exists without executable." >&2
    exit 5
fi
state_snapshotted=true

maybe_fail_publication "before-runtime-publish"

if [[ "$previous_runtime_kind" == "legacy_directory" ]]; then
    legacy_runtime_backup="${runtime_path}.legacy-backup.$$"
    if [[ -e "$legacy_runtime_backup" || -L "$legacy_runtime_backup" ]]; then
        echo "Refusing to overwrite existing runtime backup: $legacy_runtime_backup" >&2
        exit 5
    fi
    mv -- "$runtime_path" "$legacy_runtime_backup"
    legacy_runtime_moved=true
fi

runtime_publish_link="$output_parent/.manus_runtime_${mode}.publish.$$.$RANDOM"
relative_runtime_target="$(realpath --relative-to="$output_parent" \
    "$published_bundle/manus_runtime_${mode}")"
ln -s -- "$relative_runtime_target" "$runtime_publish_link"
mv -Tf -- "$runtime_publish_link" "$runtime_path"
runtime_publish_link=""
runtime_published=true
maybe_fail_publication "after-runtime-publish"

publish_executable="$output_parent/.${output_name}.publish.$$.$RANDOM"
mv -- "$staged_output" "$publish_executable"
if ! verify_bridge "$publish_executable"; then
    echo "Published-path MANUS $mode bridge verification failed before replacement." >&2
    exit 5
fi
mv -Tf -- "$publish_executable" "$output"
publish_executable=""
executable_published=true
maybe_fail_publication "after-executable-publish"

if ! verify_bridge "$output"; then
    echo "Published MANUS $mode bridge failed final verification." >&2
    exit 5
fi
maybe_fail_publication "final-verification"

publish_committed=true
bundle_to_cleanup=""
if [[ -n "$previous_executable_backup" ]]; then
    rm -f -- "$previous_executable_backup" || true
    previous_executable_backup=""
fi
if [[ -n "$old_bundle" ]]; then
    if bundle_is_owned "$old_bundle"; then
        rm -rf -- "$old_bundle" || true
    fi
fi
if [[ -n "$legacy_runtime_backup" ]]; then
    rm -rf -- "$legacy_runtime_backup" || true
    legacy_runtime_backup=""
fi

trap - EXIT
echo "Built and transactionally published MANUS bridge: $output"
echo "Bridge bundle: $published_bundle"
echo "Bridge SHA-256: $artifact_sha256"
echo "Bridge mode: $mode"
echo "Verified SDK layout: $sdk_layout"
echo "SDK library: $sdk_lib"
if [[ "$mode" == "remote" ]]; then
    echo "Strictly verified Remote runtime: $remote_runtime_prefix"
fi
env -u LD_LIBRARY_PATH -u LD_PRELOAD ldd -r "$output"
