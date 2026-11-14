# Releasing pysndlib

pysndlib's version number comes from git tags (via [setuptools-scm](https://pypi.org/project/setuptools-scm/)).
There is no version number in any source file to update — creating and pushing a tag IS the
release.

## Before you tag

1. The test workflow is green: it builds wheels for macOS (universal2), Linux (x86-64) and
   Windows (x64), Python 3.10-3.14, and runs the test suite against every wheel.
2. `CHANGELOG.md` has a new `## X.Y.Z` section at the top describing the release.

## Choosing a version (semver)

- **MAJOR** (X): breaking API changes
- **MINOR** (Y): new features, instruments, generators (most releases)
- **PATCH** (Z): bug fixes only

## Releasing from the command line

    git checkout main
    git pull
    git tag -a v0.9.0 -m "pysndlib 0.9.0"
    git push origin v0.9.0

Pushing the tag triggers the publish workflow: it builds the wheels and sdist, runs the test
suite against each wheel, and uploads everything to PyPI.

## Releasing from the GitHub web UI

GitHub's web UI creates tags through the Releases page:

1. On the repository page, click **Releases** (right sidebar) → **Draft a new release**.
2. Click **Choose a tag** and type the new tag name: `v0.9.0`.
3. In **Target**, select the branch or commit to release from (usually `main`).
4. Title it `pysndlib 0.9.0` and paste the CHANGELOG section as the description.
5. Click **Publish release**.

Publishing creates the tag, which fires the publish workflow exactly like a pushed tag would.
**Save draft** does NOT create the tag and triggers nothing until you publish — useful for
preparing the release notes in advance.

## Notes and cautions

- Between releases, test-workflow builds self-identify as `X.Y.1.devN+g<hash>` versions;
  those are never published.
- The tag is the version: do not move or delete a published tag. Wheels built from it may
  already be on PyPI, where versions are immutable. If a release goes wrong, fix forward
  and cut the next version.
- Pre-release / release-candidate versions: tag `v0.9.0rc1`. Use the **Set as a pre-release**
  checkbox in the GitHub UI; `pip` will not install `rc` versions unless asked
  (`pip install --pre`), so they are safe to publish.
