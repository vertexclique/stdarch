#!/bin/sh

# Builds documentation for all target triples that we have a registered URL for
# in liblibc. This scrapes the list of triples to document from `src/lib.rs`
# which has a bunch of `html_root_url` directives we pick up.

set -ex

rm -rf target/doc
mkdir -p target/doc

dox() {
  local arch=$1
  local target=$2

  echo documenting $arch

  if [ "$CI" != "" ]; then
    rustup target add $target || true
  fi

  rm -rf target/doc/$arch
  mkdir target/doc/$arch

  rustdoc --target $target -o target/doc/$arch src/lib.rs --crate-name simd
}

dox i686 i686-unknown-linux-gnu
dox x86_64 x86_64-unknown-linux-gnu
dox arm armv7-unknown-linux-gnueabihf

# If we're on travis, not a PR, and on the right branch, publish!
if [ "$TRAVIS_PULL_REQUEST" = "false" ] && [ "$TRAVIS_BRANCH" = "master" ]; then
  pip install ghp_import --install-option="--prefix=$HOME/.local"
  $HOME/.local/bin/ghp-import -n target/doc
  git push -qf https://${GH_TOKEN}@github.com/${TRAVIS_REPO_SLUG}.git gh-pages
fi
