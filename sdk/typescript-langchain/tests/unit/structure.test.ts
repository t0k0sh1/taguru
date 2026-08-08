/**
 * `structure.ts`'s bounded unzip — the decompression-bomb guard shared by
 * the DOCX/PPTX connectors. The central test: a zip whose HEADERS declare
 * tiny sizes while the deflate streams still expand past the cap must be
 * refused, because header sizes are attacker-forgeable (the exact bypass
 * the Python twin's `_structure.decompressed_size_within` closed).
 */

import { Unzip, UnzipInflate, zipSync } from "fflate";
import { describe, expect, it } from "vitest";

import {
  DecompressedSizeExceededError,
  unzipWithinCap,
} from "../../src/ingest-connectors/structure.js";

const deps = { Unzip, UnzipInflate };

/**
 * Overwrite the uncompressed-size field of every local file header
 * (offset 22 past `PK\x03\x04`) and central-directory entry (offset 24
 * past `PK\x01\x02`) with a small lie — the classic zip-bomb forgery.
 */
function forgeDeclaredSizes(zip: Uint8Array, forged: number): Uint8Array {
  const out = new Uint8Array(zip);
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
  // 28 covers the widest write below (central-directory offset 24 + 4);
  // a signature closer to the end has no complete size field to forge.
  for (let i = 0; i + 28 <= out.length; i += 1) {
    if (out[i] === 0x50 && out[i + 1] === 0x4b) {
      if (out[i + 2] === 0x03 && out[i + 3] === 0x04) {
        view.setUint32(i + 22, forged, true);
      } else if (out[i + 2] === 0x01 && out[i + 3] === 0x02) {
        view.setUint32(i + 24, forged, true);
      }
    }
  }
  return out;
}

describe("unzipWithinCap", () => {
  it("returns the entries of a package within the cap", () => {
    const zip = zipSync({ "word/document.xml": new TextEncoder().encode("<w:document/>") });
    const entries = unzipWithinCap(deps, zip, 1024);
    expect(new TextDecoder().decode(entries["word/document.xml"])).toBe("<w:document/>");
  });

  it("refuses a bomb whose headers declare forged small sizes", () => {
    // 200 KB of zeros deflates to a few hundred bytes; the forged headers
    // claim 10 decompressed bytes. A declared-size check would wave this
    // through — only measuring the real inflated output catches it.
    const bomb = forgeDeclaredSizes(zipSync({ "a.bin": new Uint8Array(200_000) }), 10);
    expect(() => unzipWithinCap(deps, bomb, 50_000)).toThrow(DecompressedSizeExceededError);
  });

  it("counts the total across entries, not each entry alone", () => {
    const zip = zipSync({
      "a.bin": new Uint8Array(30_000),
      "b.bin": new Uint8Array(30_000),
    });
    expect(() => unzipWithinCap(deps, zip, 40_000)).toThrow(DecompressedSizeExceededError);
    expect(Object.keys(unzipWithinCap(deps, zip, 100_000)).sort()).toEqual(["a.bin", "b.bin"]);
  });

  it("propagates a non-zip failure for the caller's corrupt path", () => {
    const garbage = new TextEncoder().encode("this is not a zip archive at all");
    expect(() => unzipWithinCap(deps, garbage, 1024)).toThrow();
    expect(() => unzipWithinCap(deps, garbage, 1024)).not.toThrow(DecompressedSizeExceededError);
  });
});
