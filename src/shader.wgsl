struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.uv = model.uv;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}


const COLOR_MAP: array<vec3<f32>, 5> = array(
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 1.0, 1.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0),
);
// Fragment shader

fn linear_srgb_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;

    let l_ = pow(l, 1.0 / 3.0);
    let m_ = pow(m, 1.0 / 3.0);
    let s_ = pow(s, 1.0 / 3.0);

    return vec3<f32>(
      0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
      1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
      0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    );
}

          // Oklab to RGB conversion
fn oklab_to_linear_srgb(c: vec3<f32>) -> vec3<f32> {
    let l_ = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
    let m_ = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
    let s_ = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    return vec3<f32>(
      4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    );
}

fn lookup_color(t: f32) -> vec3<f32> {
    let index_f = clamp(t * 4.0, 0.0, 4.0);
    let mix_f = index_f - floor(index_f);
    let index = i32(floor(index_f));
    let index_2 = clamp(index + 1, 0, 4);

    let color1_oklab = linear_srgb_to_oklab(COLOR_MAP[index]);
    let color2_oklab = linear_srgb_to_oklab(COLOR_MAP[index_2]);
    let mixed_oklab = mix(color1_oklab, color2_oklab, mix_f);

    return oklab_to_linear_srgb(mixed_oklab);
}

struct Params {
    min_temp: f32,
    max_temp: f32,
}

@group(0) @binding(0) var heat_map: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params ;


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = textureDimensions(heat_map);
    let coord = vec2<i32>(in.uv * vec2<f32>(dimensions));
    let heat = textureLoad(heat_map, coord, 0).r;
    let heat_normalized = clamp((heat - params.min_temp) / (params.max_temp - params.min_temp), 0, 1);

    let color = lookup_color(heat_normalized);

    return vec4<f32>(color, 1.0);
}
