{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f system);
    in
    {
      devShells = forEachSystem (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {

            shellHook = ''
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${
                pkgs.lib.makeLibraryPath [
                  pkgs.wayland
                  pkgs.libxkbcommon
                  pkgs.vulkan-loader
                ]
              }"
            '';
            packages = with pkgs; [
              wgsl-analyzer
              rustfmt
              rustc
              cargo
              cargo-deny
              cargo-edit
              cargo-watch
              rust-analyzer
              rustPlatform.bindgenHook
              wayland
              libxkbcommon

              vulkan-loader
              pkg-config
            ];
          };
        }
      );
    };
}
