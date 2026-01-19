@group(0) @binding(0) var heat_in: texture_2d<f32>;
@group(0) @binding(1) var heat_out: texture_storage_2d<r32float, write>;

struct Params {
    width: u32,
    height: u32,
    lambda: f32,
}

@group(0) @binding(2)
var<uniform> params: Params;

fn pos(x: i32, y: i32) -> vec2<i32> {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);

    return vec2<i32>(cx, cy);
}


@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    if x >= i32(params.width) || y >= i32(params.height) { return; }

    let c = textureLoad(heat_in, pos(x, y), 0).r;
    let n = textureLoad(heat_in, pos(x, y - 1), 0).r;
    let s = textureLoad(heat_in, pos(x, y + 1), 0).r;
    let e = textureLoad(heat_in, pos(x + 1, y), 0).r;
    let w = textureLoad(heat_in, pos(x - 1, y), 0).r;

    let laplacian = n + s + e + w - 4.0 * c;
    let new_value = c + params.lambda * laplacian;

    textureStore(heat_out, vec2<i32>(x, y), vec4<f32>(new_value, 0.0, 0.0, 0.0));
}
