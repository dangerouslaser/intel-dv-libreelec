// Compiles shaders/*.comp to SPIR-V with the system glslc (shaderc).
use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let glslc = env::var("GLSLC").unwrap_or_else(|_| "glslc".into());
    println!("cargo:rerun-if-env-changed=GLSLC");
    println!("cargo:rerun-if-changed=shaders");
    let mut names = Vec::new();
    for entry in fs::read_dir("shaders").unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("comp") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap().to_owned();
        let spv = out.join(format!("{stem}.spv"));
        let status = Command::new(&glslc)
            .args(["--target-env=vulkan1.3", "-O", "-Werror", "-Ishaders/common"])
            .arg(&path)
            .arg("-o")
            .arg(&spv)
            .status()
            .expect("run glslc (shaderc)");
        assert!(status.success(), "glslc failed on {}", path.display());
        names.push(stem);
    }
    names.sort();
    let mut table = String::from("pub static SHADERS: &[(&str, &[u8])] = &[\n");
    for n in &names {
        table.push_str(&format!(
            "    (\"{n}\", include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{n}.spv\"))),\n"
        ));
    }
    table.push_str("];\n");
    fs::write(out.join("shaders.rs"), table).unwrap();
    // yblod's host code (tables, metadata, packets) and yblod/plan.c, from
    // yblod-engine/src next to this crate (the LibreELEC package passes it).
    let yblod = PathBuf::from(env::var("YBLOD_ENGINE_SRC").expect("YBLOD_ENGINE_SRC: yblod-engine/src, patched"));
    println!("cargo:rerun-if-env-changed=YBLOD_ENGINE_SRC");
    println!("cargo:rerun-if-changed=yblod");
    let mut c = cc::Build::new();
    c.std("gnu11")
        .flag("-ffp-contract=off")
        .define("DV_FFMPEG_RAW_EXTENSIONS", None)
        .define("DV_PREPARED_CHROMA", None)
        .define("DV_ENCODE_STEPS", "256")
        .define("DV_COLOUR_LUT", None)
        .include(&yblod)
        .file("yblod/plan.c")
        .warnings(false);
    for f in ["dv_ffmpeg_metadata.c", "dv_cm4.c", "single_reshape.c", "single_reshape_ffmpeg.c", "mmr_prepare.c",
              "pixel_luma.c", "pixel_chroma_final.c", "pixel_spatial.c", "source_geometry.c",
              "source_overlay.c", "pixel_colour.c", "colour_lut.c"] {
        c.file(yblod.join(f));
    }
    if let Ok(inc) = env::var("FFMPEG_INCLUDE") {
        c.include(inc);
    }
    c.compile("yblod_host");
    // The C ABI's soname (Kodi and mpv-yblod dlopen libdv_engine_vk.so.0).
    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libdv_engine_vk.so.0");
    // libva for surface export (interop/va.rs); libgbm only for the test tools.
    println!("cargo:rustc-link-lib=va");
    println!("cargo:rustc-link-lib=va-drm");
}
