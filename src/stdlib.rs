use vm::CompileSourceFileOptions;

const RSS_MODULES: &[(&str, &str)] = &[
    (
        "stdlib/rss/bytes.rss",
        include_str!("../stdlib/rss/bytes.rss"),
    ),
    (
        "stdlib/rss/collections.rss",
        include_str!("../stdlib/rss/collections.rss"),
    ),
    ("stdlib/rss/cli.rss", include_str!("../stdlib/rss/cli.rss")),
    ("stdlib/rss/io.rss", include_str!("../stdlib/rss/io.rss")),
    (
        "stdlib/rss/iter.rss",
        include_str!("../stdlib/rss/iter.rss"),
    ),
    (
        "stdlib/rss/lrucache.rss",
        include_str!("../stdlib/rss/lrucache.rss"),
    ),
    (
        "stdlib/rss/math.rss",
        include_str!("../stdlib/rss/math.rss"),
    ),
    (
        "stdlib/rss/parse.rss",
        include_str!("../stdlib/rss/parse.rss"),
    ),
    (
        "stdlib/rss/path.rss",
        include_str!("../stdlib/rss/path.rss"),
    ),
    (
        "stdlib/rss/strings.rss",
        include_str!("../stdlib/rss/strings.rss"),
    ),
];

pub(crate) fn compile_options() -> CompileSourceFileOptions {
    let mut options = CompileSourceFileOptions::new();
    for (spec, source) in RSS_MODULES {
        options.set_module_override_source(*spec, *source);
    }
    options
}
