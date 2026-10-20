# Camera palette reference

[Imaging guide](imaging-guide.md)

## Custom palette files

The Imaging panel's collapsible **Palette builder** creates named gradients with
2–64 RGB color stops. Colors interpolate linearly in RGB into a cached 256-entry
display lookup table. Positions run from 0 (cool) to 1 (hot), must increase strictly,
and must include both endpoints. Names contain 1–100 printable characters and are
unique without regard to case. The local library is `palettes.json` beside the
application preferences; it is separate from bundled pipeline presets.

Import accepts the following versioned JSON format (up to 128 palettes, 2 MiB).
Export writes one current palette using the same format:

```json
{
  "version": 1,
  "palettes": [
    {
      "name": "Blue to red",
      "stops": [
        {"position": 0, "color": "#0000ff"},
        {"position": 0.5, "color": "#ffffff"},
        {"position": 1, "color": "#ff0000"}
      ]
    }
  ]
}
```

Invalid files are rejected as a whole. Imports merge the library and ask before
replacing matching names. Existing pipeline selections retain their embedded
snapshot when importing or deleting library entries. An embedded selection absent
from the library is labeled **Embedded** in the palette menu.

Pipeline palette strings beginning with `custom:` contain the complete named
gradient as JSON. They require no library lookup during processing. Ordinary
built-in palette identifiers and hardware palette IDs retain their existing meaning.

## Palette IDs and labels

Palettes 10–22 have descriptive names based on sampled TC002C Duo output.
These are app labels rather than official vendor names; their camera IDs are
unchanged. White hot (1) and Black hot (2) retain their existing labels.

| Camera ID | Name            | Sampled colors, cool → warm                                 |
| --------- | --------------- | ----------------------------------------------------------- |
| 10        | Violet iron     | Dark blue, violet, magenta, orange, yellow                  |
| 11        | Classic rainbow | Dark violet, blue, green, orange, cream                     |
| 12        | Fire            | Black, dark red, red, orange, yellow, cream                 |
| 13        | Sunset          | Purple, magenta, red, orange, pale yellow                   |
| 14        | Soft iron       | Dark blue, muted violet, magenta, orange, pale yellow       |
| 15        | Amber           | Black, brown, amber, yellow, pale gold                      |
| 16        | Vivid rainbow   | Bright magenta, violet, blue, cyan, green, yellow, orange   |
| 17        | Neon sunset     | Blue, violet, bright magenta, red, orange, yellow           |
| 18        | Silver          | Black, gray, silver, white                                  |
| 19        | Pastel rainbow  | Dark violet, magenta, blue, green, yellow, peach, pale pink |
| 20        | Red hot         | Black, dark red, bright red                                 |
| 21        | Green hot       | Black, dark green, bright green                             |
| 22        | Ocean heat      | Dark blue, blue, cyan, green, yellow, pale highlights       |
