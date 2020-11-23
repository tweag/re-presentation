{ pkgs ? import <nixpkgs> {} }:
let
  mytexlive = pkgs.texlive.combine {inherit (pkgs.texlive) scheme-medium pgf xetex fontspec euenc enumitem pgfplots adjustbox collectbox tcolorbox environ trimspaces tikz-cd; };
in
  pkgs.mkShell {
    FONTCONFIG_FILE = pkgs.makeFontsConf { fontDirectories = [ pkgs.google-fonts ]; };
    buildInputs = [ mytexlive ];
}
