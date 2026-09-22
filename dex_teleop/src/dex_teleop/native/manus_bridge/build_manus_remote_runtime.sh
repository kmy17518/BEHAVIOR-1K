#!/usr/bin/env bash
set -euo pipefail

readonly runtime_schema="2"
readonly runtime_kind="dex_teleop_manus_remote_runtime"
readonly marker_name=".dex-teleop-manus-remote-runtime"
readonly grpc_tag="v1.28.1"
readonly grpc_commit="cb81fe0dfaa424eb50de26fb7c904a27a78c3f76"
readonly protobuf_commit="fe1790ca0df67173702f70d5646b82f48f412b99"
readonly empty_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
readonly sdk_header_sha256="0beeec96f323c0bc87df707940f379f3dca0ec5b31bea9e49b90979ad7bb5b62"
readonly sdk_types_sha256="f20bfe782e043c8aaa2f7e89eee6df6b701ecb7711150f7ab3d7a1a66d86d892"
readonly sdk_initializers_sha256="4e9c6607138ece3351eaf072634920ba1b16bde89483151c50700713745062e9"
readonly sdk_remote_sha256="91b42c423e36031c7bade01964c341c9f9423b28aeadd300056792d982c0ccaf"
readonly sdk_integrated_sha256="0e67141b97b64c089c3bbdab47980ca9822c4de19adea810a2f68722adcb3fe3"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/dex_teleop"
prefix="${DEX_TELEOP_MANUS_REMOTE_RUNTIME:-$cache_root/manus_sdk_v3.1.1_remote_runtime}"
source_dir="$cache_root/sources/grpc-v1.28.1"
jobs="$(nproc)"
force=false
allow_owned_prefix_outside_cache=false
validate_prefix_only=false
validate_source_only=false
verify_only=false

usage() {
    cat <<'EOF'
Usage: build_manus_remote_runtime.sh [OPTIONS]

Build the MANUS SDK v3.1.1 Remote runtime dependencies into a user-writable,
script-owned prefix. This follows the release's SDKClient_Linux/Dockerfile:
gRPC v1.28.1 and its pinned Protobuf v3.11.2 submodule, with Protobuf
configured --enable-shared and its test suite run before gRPC is built.

Options:
  --prefix PATH       Runtime prefix. Defaults below.
  --source-dir PATH   Pinned gRPC object/cache checkout.
  --jobs N            Parallel build jobs.
  --force             Replace an existing runtime only when script ownership
                      is proven. The previous runtime is restored on failure.
  --allow-owned-prefix-outside-cache
                      Permit a non-system, non-repository prefix outside the
                      approved cache only when its ownership marker is valid.
  --verify-only       Strictly validate an existing runtime and exit.
  --validate-prefix-only
                      Validate path/destructive safety without building.
  --validate-source-only
                      Create and validate a disposable pinned source worktree,
                      then remove it without building.

Default prefix:
  ${XDG_CACHE_HOME:-$HOME/.cache}/dex_teleop/manus_sdk_v3.1.1_remote_runtime

No sudo, package-manager, system ldconfig, global symlink, system library, or
Conda library operation is performed.
EOF
}

while (($#)); do
    case "$1" in
        --prefix)
            [[ $# -ge 2 ]] || { echo "--prefix requires a path" >&2; exit 2; }
            prefix="$2"
            shift 2
            ;;
        --source-dir)
            [[ $# -ge 2 ]] || { echo "--source-dir requires a path" >&2; exit 2; }
            source_dir="$2"
            shift 2
            ;;
        --jobs)
            [[ $# -ge 2 ]] || { echo "--jobs requires a positive integer" >&2; exit 2; }
            jobs="$2"
            shift 2
            ;;
        --force)
            force=true
            shift
            ;;
        --allow-owned-prefix-outside-cache)
            allow_owned_prefix_outside_cache=true
            shift
            ;;
        --verify-only)
            verify_only=true
            shift
            ;;
        --validate-prefix-only)
            validate_prefix_only=true
            shift
            ;;
        --validate-source-only)
            validate_source_only=true
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

if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "--jobs requires a positive integer, got: $jobs" >&2
    exit 2
fi
if [[ "$prefix" == *$'\n'* || "$source_dir" == *$'\n'* ]]; then
    echo "Paths containing newlines are not supported." >&2
    exit 2
fi

for tool in git make gcc g++ autoconf automake libtoolize pkg-config readelf \
    realpath sha256sum awk ldd readlink; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required build tool is unavailable: $tool" >&2
        exit 3
    fi
done

cache_root="$(realpath -m -- "$cache_root")"
prefix="$(realpath -m -- "$prefix")"
source_dir="$(realpath -m -- "$source_dir")"
readonly approved_default_prefix="$cache_root/manus_sdk_v3.1.1_remote_runtime"
readonly approved_prefix_pool="$cache_root/manus_remote_runtimes"

path_is_at_or_below() {
    local path="$1"
    local root="$2"
    [[ "$path" == "$root" || "$path" == "$root/"* ]]
}

metadata_value() {
    local key="$1"
    local file="$2"
    awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$file" 2>/dev/null || true
}

file_sha256() {
    sha256sum -- "$1" | awk '{ print $1 }'
}

library_soname() {
    readelf -d "$(readlink -f -- "$1")" 2>/dev/null |
        awk -F'[][]' '/\(SONAME\)/ { print $2; exit }'
}

pkg_version() {
    awk '$1 == "Version:" { print $2; exit }' "$1" 2>/dev/null || true
}

write_ownership_marker() {
    local runtime_prefix="$1"
    cat > "$runtime_prefix/$marker_name" <<EOF
kind=$runtime_kind
schema=$runtime_schema
canonical_prefix=$runtime_prefix
EOF
}

ownership_marker_is_valid() {
    local runtime_prefix="$1"
    local marker="$runtime_prefix/$marker_name"
    [[ -f "$marker" ]] || return 1
    [[ "$(metadata_value kind "$marker")" == "$runtime_kind" ]] || return 1
    [[ "$(metadata_value schema "$marker")" == "$runtime_schema" ]] || return 1
    [[ "$(metadata_value canonical_prefix "$marker")" == "$runtime_prefix" ]] || return 1
}

validate_prefix_safety() {
    local candidate="$1"
    local approved=false
    if [[ "$candidate" == "$approved_default_prefix" ||
          ( "$candidate" == "$approved_prefix_pool/"* &&
            "$candidate" != "$approved_prefix_pool" ) ]]; then
        approved=true
    fi

    local repo_root=""
    repo_root="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || true)"
    [[ -z "$repo_root" ]] || repo_root="$(realpath -m -- "$repo_root")"

    local home_root
    home_root="$(realpath -m -- "$HOME")"
    if [[ "$candidate" == "/" || "$candidate" == "$home_root" ]]; then
        echo "Refusing unsafe runtime prefix: $candidate" >&2
        return 1
    fi

    local forbidden_subtrees=(
        "/bin"
        "/boot"
        "/dev"
        "/etc"
        "/lib"
        "/lib64"
        "/opt"
        "/proc"
        "/run"
        "/sbin"
        "/sys"
        "/usr"
        "/usr/local"
        "/var"
        "$(realpath -m -- "$HOME/manus_setup")"
        "$(realpath -m -- "$HOME/Desktop/emg/manus")"
        "$(realpath -m -- "$HOME/miniconda3")"
        "$(realpath -m -- "$HOME/anaconda3")"
    )
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        forbidden_subtrees+=("$(realpath -m -- "$CONDA_PREFIX")")
    fi
    if [[ -n "$repo_root" ]]; then
        forbidden_subtrees+=("$repo_root")
    fi

    local forbidden
    for forbidden in "${forbidden_subtrees[@]}"; do
        if path_is_at_or_below "$candidate" "$forbidden"; then
            echo "Refusing runtime prefix inside protected path $forbidden: $candidate" >&2
            return 1
        fi
    done
    if path_is_at_or_below "$candidate" "$cache_root" && [[ "$approved" == false ]]; then
        echo "Refusing broad or unapproved dex_teleop cache prefix: $candidate" >&2
        return 1
    fi
    if [[ "$approved" == true ]]; then
        return 0
    fi
    if [[ "$allow_owned_prefix_outside_cache" != true ]]; then
        echo "Prefix is outside the approved MANUS cache: $candidate" >&2
        echo "Use an approved cache path, or the explicit owned-prefix override for an existing marked runtime." >&2
        return 1
    fi
    if ! ownership_marker_is_valid "$candidate"; then
        echo "Outside-cache prefix lacks a valid script ownership marker: $candidate" >&2
        return 1
    fi
}

expected_manifest_files() {
    printf '%s\n' \
        "$marker_name" \
        "BUILD-METADATA.txt" \
        "bin/protoc" \
        "include/google/protobuf/message.h" \
        "include/grpc/grpc.h" \
        "lib/libgrpc.so.9" \
        "lib/libgrpc.so.9.0.0" \
        "lib/libgrpc++.so.1" \
        "lib/libgrpc++.so.1.28.1" \
        "lib/libprotobuf.so.22" \
        "lib/libprotobuf.so.22.0.2" \
        "lib/pkgconfig/grpc.pc" \
        "lib/pkgconfig/grpc++.pc" \
        "lib/pkgconfig/protobuf.pc"
}

validate_library_link() {
    local runtime_prefix="$1"
    local link_name="$2"
    local target_name="$3"
    local expected_soname="$4"
    local link_path="$runtime_prefix/lib/$link_name"
    [[ -L "$link_path" ]] || return 1
    [[ "$(readlink -- "$link_path")" == "$target_name" ]] || return 1
    local target_path
    target_path="$(readlink -f -- "$link_path")"
    [[ "$target_path" == "$runtime_prefix/lib/$target_name" ]] || return 1
    [[ -f "$target_path" ]] || return 1
    [[ "$(library_soname "$target_path")" == "$expected_soname" ]] || return 1
    local elf_header
    elf_header="$(readelf -h "$target_path")"
    [[ "$elf_header" == *"Class:                             ELF64"* ]] || return 1
    [[ "$elf_header" == *"Machine:                           Advanced Micro Devices X86-64"* ]] || return 1
}

validate_load_closure() {
    local runtime_prefix="$1"
    local library
    for library in libgrpc.so.9 libgrpc++.so.1 libprotobuf.so.22; do
        local output
        output="$(env -u LD_PRELOAD LD_LIBRARY_PATH="$runtime_prefix/lib" \
            ldd -r "$runtime_prefix/lib/$library" 2>&1 || true)"
        if rg='not found|undefined symbol'; [[ "$output" =~ $rg ]]; then
            echo "Runtime load closure failed for $library:" >&2
            printf '%s\n' "$output" >&2
            return 1
        fi
    done
}

source_status_snapshot() {
    local source_root="$1"
    git -C "$source_root" status --porcelain=v1 --untracked-files=all
    git -C "$source_root" submodule foreach --quiet --recursive '
        status="$(git status --porcelain=v1 --untracked-files=all)" || exit
        if test -n "$status"; then
            printf "submodule:%s\n%s\n" "$displaypath" "$status"
        fi
    '
}

verify_runtime() {
    local runtime_prefix="$1"
    local lib_dir="$runtime_prefix/lib"
    local metadata="$runtime_prefix/BUILD-METADATA.txt"
    local manifest="$runtime_prefix/MANIFEST.sha256"
    ownership_marker_is_valid "$runtime_prefix" || return 1
    [[ -f "$metadata" && -f "$manifest" ]] || return 1

    [[ "$(metadata_value manifest_schema "$metadata")" == "$runtime_schema" ]] || return 1
    [[ "$(metadata_value runtime_kind "$metadata")" == "$runtime_kind" ]] || return 1
    [[ "$(metadata_value install_prefix "$metadata")" == "$runtime_prefix" ]] || return 1
    [[ "$(metadata_value grpc_tag "$metadata")" == "$grpc_tag" ]] || return 1
    [[ "$(metadata_value grpc_commit "$metadata")" == "$grpc_commit" ]] || return 1
    [[ "$(metadata_value grpc_tag_commit "$metadata")" == "$grpc_commit" ]] || return 1
    [[ "$(metadata_value protobuf_commit "$metadata")" == "$protobuf_commit" ]] || return 1
    [[ "$(metadata_value source_disposable_worktree "$metadata")" == "true" ]] || return 1
    [[ "$(metadata_value source_prebuild_dirty "$metadata")" == "false" ]] || return 1
    [[ "$(metadata_value source_prebuild_status_sha256 "$metadata")" == "$empty_sha256" ]] || return 1
    local postbuild_dirty postbuild_status_sha256 submodule_manifest_sha256
    postbuild_dirty="$(metadata_value source_postbuild_dirty "$metadata")"
    postbuild_status_sha256="$(metadata_value source_postbuild_status_sha256 "$metadata")"
    submodule_manifest_sha256="$(metadata_value source_submodule_manifest_sha256 "$metadata")"
    [[ "$postbuild_dirty" == "true" || "$postbuild_dirty" == "false" ]] || return 1
    [[ "$postbuild_status_sha256" =~ ^[0-9a-f]{64}$ ]] || return 1
    [[ "$submodule_manifest_sha256" =~ ^[0-9a-f]{64}$ ]] || return 1
    [[ "$(metadata_value sdk_header_sha256 "$metadata")" == "$sdk_header_sha256" ]] || return 1
    [[ "$(metadata_value sdk_types_sha256 "$metadata")" == "$sdk_types_sha256" ]] || return 1
    [[ "$(metadata_value sdk_initializers_sha256 "$metadata")" == "$sdk_initializers_sha256" ]] || return 1
    [[ "$(metadata_value sdk_remote_sha256 "$metadata")" == "$sdk_remote_sha256" ]] || return 1
    [[ "$(metadata_value sdk_integrated_sha256 "$metadata")" == "$sdk_integrated_sha256" ]] || return 1

    [[ "$(pkg_version "$lib_dir/pkgconfig/grpc.pc")" == "9.0.0" ]] || return 1
    [[ "$(pkg_version "$lib_dir/pkgconfig/grpc++.pc")" == "1.28.1" ]] || return 1
    [[ "$(pkg_version "$lib_dir/pkgconfig/protobuf.pc")" == "3.11.2" ]] || return 1
    [[ "$(env LD_LIBRARY_PATH="$lib_dir" "$runtime_prefix/bin/protoc" --version)" == "libprotoc 3.11.2" ]] || return 1
    validate_library_link "$runtime_prefix" "libgrpc.so.9" "libgrpc.so.9.0.0" "libgrpc.so.9" || return 1
    validate_library_link "$runtime_prefix" "libgrpc++.so.1" "libgrpc++.so.1.28.1" "libgrpc++.so.1" || return 1
    validate_library_link "$runtime_prefix" "libprotobuf.so.22" "libprotobuf.so.22.0.2" "libprotobuf.so.22" || return 1

    local expected_paths actual_paths
    expected_paths="$(expected_manifest_files | sort)"
    actual_paths="$(awk '{ print $2 }' "$manifest" | sort)"
    [[ "$actual_paths" == "$expected_paths" ]] || return 1
    (cd "$runtime_prefix" && sha256sum --quiet -c MANIFEST.sha256) || return 1
    [[ "$(metadata_value sha256_libgrpc "$metadata")" == \
       "$(file_sha256 "$lib_dir/libgrpc.so.9.0.0")" ]] || return 1
    [[ "$(metadata_value sha256_libgrpcxx "$metadata")" == \
       "$(file_sha256 "$lib_dir/libgrpc++.so.1.28.1")" ]] || return 1
    [[ "$(metadata_value sha256_libprotobuf "$metadata")" == \
       "$(file_sha256 "$lib_dir/libprotobuf.so.22.0.2")" ]] || return 1
    [[ "$(metadata_value sha256_protoc "$metadata")" == \
       "$(file_sha256 "$runtime_prefix/bin/protoc")" ]] || return 1
    validate_load_closure "$runtime_prefix" || return 1
}

legacy_default_runtime_is_owned() {
    local runtime_prefix="$1"
    local metadata="$runtime_prefix/BUILD-METADATA.txt"
    [[ "$runtime_prefix" == "$approved_default_prefix" ]] || return 1
    [[ -f "$metadata" ]] || return 1
    [[ "$(awk -F': ' '$1 == "MANUS SDK compatibility target" { print $2 }' "$metadata")" == "3.1.1 Remote" ]] || return 1
    [[ "$(awk -F': ' '$1 == "gRPC commit" { print $2 }' "$metadata")" == "$grpc_commit" ]] || return 1
    [[ "$(awk -F': ' '$1 == "Protobuf commit" { print $2 }' "$metadata")" == "$protobuf_commit" ]] || return 1
    validate_library_link "$runtime_prefix" "libgrpc.so.9" "libgrpc.so.9.0.0" "libgrpc.so.9" || return 1
    validate_library_link "$runtime_prefix" "libgrpc++.so.1" "libgrpc++.so.1.28.1" "libgrpc++.so.1" || return 1
    validate_library_link "$runtime_prefix" "libprotobuf.so.22" "libprotobuf.so.22.0.2" "libprotobuf.so.22" || return 1
    validate_load_closure "$runtime_prefix" || return 1
}

if ! validate_prefix_safety "$prefix"; then
    exit 2
fi

existing_owned=false
existing_legacy_owned=false
if [[ -e "$prefix" || -L "$prefix" ]]; then
    if ownership_marker_is_valid "$prefix"; then
        existing_owned=true
    elif legacy_default_runtime_is_owned "$prefix"; then
        existing_legacy_owned=true
    fi
fi

if [[ "$validate_prefix_only" == true ]]; then
    if [[ "$force" == true && ( -e "$prefix" || -L "$prefix" ) &&
          "$existing_owned" != true && "$existing_legacy_owned" != true ]]; then
        echo "Refusing to replace an existing prefix not owned by this script: $prefix" >&2
        exit 4
    fi
    echo "Validated MANUS Remote runtime prefix safety: $prefix"
    exit 0
fi

if [[ "$verify_only" == true ]]; then
    if verify_runtime "$prefix"; then
        echo "Verified existing MANUS Remote runtime: $prefix"
        exit 0
    fi
    echo "Existing MANUS Remote runtime failed strict manifest/ABI validation: $prefix" >&2
    exit 4
fi

if [[ "$validate_source_only" != true &&
      ( -e "$prefix" || -L "$prefix" ) && "$force" == false ]]; then
    if verify_runtime "$prefix"; then
        echo "Verified existing MANUS Remote runtime: $prefix"
        exit 0
    fi
    echo "Runtime prefix exists but failed ownership, manifest, version, or ABI validation: $prefix" >&2
    echo "Inspect it before using --force; unowned paths are never replaced." >&2
    exit 4
fi
if [[ "$validate_source_only" != true && "$force" == true &&
      ( -e "$prefix" || -L "$prefix" ) &&
      "$existing_owned" != true && "$existing_legacy_owned" != true ]]; then
    echo "Refusing to replace an existing prefix not owned by this script: $prefix" >&2
    exit 4
fi

mkdir -p -- "$(dirname -- "$source_dir")"
if [[ ! -d "$source_dir/.git" ]]; then
    if [[ -e "$source_dir" || -L "$source_dir" ]]; then
        echo "Source path exists but is not a gRPC git checkout: $source_dir" >&2
        exit 4
    fi
    git clone -b "$grpc_tag" https://github.com/grpc/grpc "$source_dir"
    git -C "$source_dir" submodule update --init --recursive
fi

origin_url="$(git -C "$source_dir" remote get-url origin 2>/dev/null || true)"
case "$origin_url" in
    https://github.com/grpc/grpc|https://github.com/grpc/grpc.git) ;;
    *)
        echo "gRPC source origin is not the vendor-declared upstream: ${origin_url:-missing}" >&2
        exit 4
        ;;
esac
actual_tag_commit="$(git -C "$source_dir" rev-parse "refs/tags/$grpc_tag^{commit}" 2>/dev/null || true)"
if [[ "$actual_tag_commit" != "$grpc_commit" ]]; then
    echo "gRPC tag $grpc_tag resolves to ${actual_tag_commit:-missing}; expected $grpc_commit." >&2
    exit 4
fi
if ! git -C "$source_dir" cat-file -e "$grpc_commit^{commit}" 2>/dev/null; then
    echo "Pinned gRPC commit is absent from source cache: $grpc_commit" >&2
    exit 4
fi

build_source_parent="$cache_root/build-sources"
mkdir -p -- "$build_source_parent"
build_source="$build_source_parent/grpc-v1.28.1.$$.$RANDOM"
if [[ -e "$build_source" || -L "$build_source" ]]; then
    echo "Disposable build source already exists unexpectedly: $build_source" >&2
    exit 4
fi

worktree_active=false
prefix_build_started=false
publish_complete=false
backup_prefix=""
backup_was_owned=false

cleanup() {
    local status=$?
    trap - EXIT
    if [[ "$worktree_active" == true ]]; then
        git -C "$source_dir" worktree remove --force "$build_source" >/dev/null 2>&1 || true
        git -C "$source_dir" worktree prune >/dev/null 2>&1 || true
    fi
    if [[ $status -ne 0 && "$prefix_build_started" == true ]]; then
        if ownership_marker_is_valid "$prefix"; then
            rm -rf -- "$prefix"
        else
            echo "Refusing cleanup of prefix after ownership marker loss: $prefix" >&2
        fi
        if [[ -n "$backup_prefix" && -e "$backup_prefix" ]]; then
            mv -- "$backup_prefix" "$prefix"
        fi
        echo "Build failed; removed only the marked incomplete runtime and restored the previous prefix." >&2
    fi
    exit "$status"
}
trap cleanup EXIT

git -C "$source_dir" worktree add --detach "$build_source" "$grpc_commit"
worktree_active=true
git -C "$build_source" submodule update --init --recursive

actual_grpc_commit="$(git -C "$build_source" rev-parse HEAD)"
actual_protobuf_commit="$(git -C "$build_source/third_party/protobuf" rev-parse HEAD)"
if [[ "$actual_grpc_commit" != "$grpc_commit" ||
      "$actual_protobuf_commit" != "$protobuf_commit" ]]; then
    echo "Disposable source commits do not match the vendor pins." >&2
    exit 4
fi
submodule_status="$(git -C "$build_source" submodule status --recursive)"
if printf '%s\n' "$submodule_status" | awk '$1 ~ /^[-+U]/ { bad = 1 } END { exit bad }'; then
    :
else
    echo "A recursive submodule does not match its pinned gitlink." >&2
    exit 4
fi
prebuild_status="$(source_status_snapshot "$build_source")"
if [[ -n "$prebuild_status" ]]; then
    echo "Disposable gRPC source or a recursive submodule is dirty before the build:" >&2
    printf '%s\n' "$prebuild_status" >&2
    exit 4
fi
prebuild_status_sha256="$(printf '%s' "$prebuild_status" | sha256sum | awk '{ print $1 }')"
submodule_manifest_sha256="$(printf '%s\n' "$submodule_status" | sha256sum | awk '{ print $1 }')"

if [[ "$validate_source_only" == true ]]; then
    echo "Validated clean disposable gRPC source: $actual_grpc_commit"
    echo "Validated pinned Protobuf source: $actual_protobuf_commit"
    exit 0
fi

if [[ -e "$prefix" || -L "$prefix" ]]; then
    backup_prefix="${prefix}.backup.$$"
    if [[ -e "$backup_prefix" || -L "$backup_prefix" ]]; then
        echo "Refusing to overwrite existing runtime backup: $backup_prefix" >&2
        exit 4
    fi
    backup_was_owned=true
    mv -- "$prefix" "$backup_prefix"
fi
mkdir -p -- "$prefix"
write_ownership_marker "$prefix"
prefix_build_started=true

protobuf_dir="$build_source/third_party/protobuf"
(
    cd "$protobuf_dir"
    ./autogen.sh
    ./configure --enable-shared --prefix="$prefix"
    make -j"$jobs"
    make -j"$jobs" check
    make install
)

(
    cd "$build_source"
    rm -f cache.mk
    ldconfig_shim_dir="$prefix/.build-tools"
    mkdir -p "$ldconfig_shim_dir"
    printf '#!/bin/sh\nexit 0\n' > "$ldconfig_shim_dir/ldconfig"
    chmod +x "$ldconfig_shim_dir/ldconfig"
    export PATH="$ldconfig_shim_dir:$prefix/bin:$PATH"
    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig"
    export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="$prefix/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    [[ "$(pkg-config --modversion protobuf)" == "3.11.2" ]]
    [[ "$(protoc --version)" == "libprotoc 3.11.2" ]]
    make -j"$jobs" HAS_SYSTEM_PROTOBUF=true HAS_SYSTEM_OPENSSL_ALPN=false
    make prefix="$prefix" HAS_SYSTEM_PROTOBUF=true HAS_SYSTEM_OPENSSL_ALPN=false install
    rm -rf -- "$ldconfig_shim_dir"
)

postbuild_status="$(source_status_snapshot "$build_source")"
postbuild_dirty=false
[[ -z "$postbuild_status" ]] || postbuild_dirty=true
postbuild_status_sha256="$(printf '%s' "$postbuild_status" | sha256sum | awk '{ print $1 }')"

libgrpc_sha256="$(file_sha256 "$prefix/lib/libgrpc.so.9.0.0")"
libgrpcxx_sha256="$(file_sha256 "$prefix/lib/libgrpc++.so.1.28.1")"
libprotobuf_sha256="$(file_sha256 "$prefix/lib/libprotobuf.so.22.0.2")"
protoc_sha256="$(file_sha256 "$prefix/bin/protoc")"

cat > "$prefix/BUILD-METADATA.txt" <<EOF
manifest_schema=$runtime_schema
runtime_kind=$runtime_kind
install_prefix=$prefix
manus_sdk_compatibility=3.1.1-remote
grpc_tag=$grpc_tag
grpc_tag_commit=$actual_tag_commit
grpc_commit=$actual_grpc_commit
grpc_core_pkgconfig_version=9.0.0
grpc_cpp_pkgconfig_version=1.28.1
protobuf_version=3.11.2
protobuf_commit=$actual_protobuf_commit
source_disposable_worktree=true
source_prebuild_dirty=false
source_prebuild_status_sha256=$prebuild_status_sha256
source_postbuild_dirty=$postbuild_dirty
source_postbuild_status_sha256=$postbuild_status_sha256
source_submodule_manifest_sha256=$submodule_manifest_sha256
protobuf_configure_flags=--enable-shared --prefix=$prefix
grpc_make_overrides=HAS_SYSTEM_PROTOBUF=true HAS_SYSTEM_OPENSSL_ALPN=false
cc_path=$(command -v gcc)
cc_version=$(gcc -dumpfullversion -dumpversion)
cxx_path=$(command -v g++)
cxx_version=$(g++ -dumpfullversion -dumpversion)
ld_version=$(ld --version | awk 'NR == 1 { print; exit }')
make_version=$(make --version | awk 'NR == 1 { print; exit }')
autoconf_version=$(autoconf --version | awk 'NR == 1 { print; exit }')
libtoolize_version=$(libtoolize --version | awk 'NR == 1 { print; exit }')
sdk_header_sha256=$sdk_header_sha256
sdk_types_sha256=$sdk_types_sha256
sdk_initializers_sha256=$sdk_initializers_sha256
sdk_remote_sha256=$sdk_remote_sha256
sdk_integrated_sha256=$sdk_integrated_sha256
sha256_libgrpc=$libgrpc_sha256
sha256_libgrpcxx=$libgrpcxx_sha256
sha256_libprotobuf=$libprotobuf_sha256
sha256_protoc=$protoc_sha256
EOF

mapfile -t manifest_files < <(expected_manifest_files)
(
    cd "$prefix"
    sha256sum -- "${manifest_files[@]}" > MANIFEST.sha256
)

if ! verify_runtime "$prefix"; then
    echo "Built runtime failed strict ownership, manifest, version, SONAME, or ABI validation." >&2
    exit 5
fi

git -C "$source_dir" worktree remove --force "$build_source"
git -C "$source_dir" worktree prune
worktree_active=false

if [[ -n "$backup_prefix" && "$backup_was_owned" == true ]]; then
    rm -rf -- "$backup_prefix"
fi
publish_complete=true
prefix_build_started=false
trap - EXIT
echo "Built and strictly verified MANUS Remote runtime: $prefix"
