{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    cargo
    rustc
    openssl
    pkg-config
  ];

  shellHook = ''
    echo "Rust dev environment loaded"
    echo "cargo $(cargo --version)"
    echo "rustc $(rustc --version)"
  '';
}
