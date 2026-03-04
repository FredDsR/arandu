"""Google Drive catalog metadata extractor.

Parses filename and gdrive_path conventions from the project's
Google Drive folder organization to extract structured interview metadata.
"""

from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath

from arandu.core.metadata.protocol import MetadataExtractor
from arandu.schemas import SourceMetadata

logger = logging.getLogger(__name__)

# Default known researcher names (case-insensitive matching)
_DEFAULT_RESEARCHERS: frozenset[str] = frozenset(
    {
        "Dani Borges",
        "DaniBorges",
        "Dani",
        "Glenio",
        "Glênio",
        "Glenio Rissio",
    }
)

# Default known location keywords (case-insensitive matching)
_DEFAULT_LOCATIONS: frozenset[str] = frozenset(
    {
        "BARRA DE PELOTAS",
        "BARRA",
        "DOQUINHAS",
        "BALSA",
    }
)

# Regex for DD-MM-YY or DD-MM-YYYY dates (trailing boundary allows non-digit like "2025VID")
_DATE_DASHED = re.compile(r"\b(\d{2}-\d{2}-\d{2,4})(?:\b|(?=\D))")
# Regex for DD.MM short dates
_DATE_DOTTED = re.compile(r"\b(\d{2}\.\d{2})\b")
# Regex for "DD de MMM." Portuguese month abbreviation dates
_DATE_WRITTEN = re.compile(r"\b(\d{1,2} de [a-zç]{3,4}\.?)", re.IGNORECASE)
# Android VID timestamp: VID_YYYYMMDD_HHMMSSMMM
_VID_TIMESTAMP = re.compile(r"^VID_(\d{4})(\d{2})(\d{2})_")
# Trailing sequence number before extension: _NN or space NN or - NN
_TRAILING_SEQ = re.compile(r"[\s_\-]+(\d{1,2})\s*$")
# Parte label: "Parte I", "Parte II", etc.
_PARTE_LABEL = re.compile(r"(Parte\s+[IVX]+)", re.IGNORECASE)
# Camera file patterns
_CAMERA_FILE = re.compile(r"^(MVI_\d+|IMG_\d+|_MG_\d+|DSC_\d+)\.\w+$", re.IGNORECASE)
# Android VID file pattern
_ANDROID_VID = re.compile(r"^VID_\d{8}_\d+", re.IGNORECASE)
# Device prefix in brackets: [Tablet]
_DEVICE_PREFIX = re.compile(r"^\[([^\]]+)\]")
# Location folder pattern: strips "- Fotos, Vídeos, Áudios" and similar suffixes
_LOCATION_SUFFIX = re.compile(r"\s*[-\u2013]\s*Fotos.*$", re.IGNORECASE)

# Path segment after IMAGENS e ÁUDIOS (the location folder)
_MEDIA_ROOT = "IMAGENS e ÁUDIOS"


class GDriveCatalogExtractor(MetadataExtractor):
    """Extract interview metadata from Google Drive catalog rows.

    Parses filename and gdrive_path columns using pattern matching
    to extract participant names, researcher names, locations, dates,
    and sequence numbers.
    """

    def __init__(
        self,
        known_researchers: set[str] | None = None,
        known_locations: set[str] | None = None,
    ) -> None:
        """Initialize the extractor with known entity sets.

        Args:
            known_researchers: Names to recognize as researchers.
                Defaults to project-known researchers.
            known_locations: Location keywords to recognize.
                Defaults to project-known locations.
        """
        researchers = known_researchers if known_researchers is not None else _DEFAULT_RESEARCHERS
        locations = known_locations if known_locations is not None else _DEFAULT_LOCATIONS

        # Build case-insensitive lookup: lowered_name -> original_name
        self._researchers: dict[str, str] = {r.lower(): r for r in researchers}
        self._locations: dict[str, str] = {loc.lower(): loc for loc in locations}

    def extract(self, row: dict[str, str]) -> SourceMetadata:
        """Extract structured metadata from a catalog row.

        Args:
            row: Dictionary of catalog column values (e.g., from CSV DictReader).

        Returns:
            SourceMetadata with extracted fields.
        """
        name = row.get("name", "")
        gdrive_path = row.get("gdrive_path", "")

        # Extract from filename
        filename_data = self._extract_from_filename(name)

        # Extract from path
        path_data = self._extract_from_path(gdrive_path)

        # Merge: filename takes priority, path fills gaps
        participant = self._sanitize_name(
            filename_data.get("participant") or path_data.get("participant")
        )
        researcher = self._sanitize_name(
            filename_data.get("researcher") or path_data.get("researcher")
        )
        location = filename_data.get("location") or path_data.get("location")
        recording_date = filename_data.get("date") or path_data.get("date")
        sequence_number = filename_data.get("sequence_number")
        sequence_label = filename_data.get("sequence_label")
        event_context = path_data.get("event_context")

        # Use created_time as fallback date
        if not recording_date and row.get("created_time"):
            recording_date = row["created_time"][:10]  # YYYY-MM-DD

        metadata = SourceMetadata(
            participant_name=participant,
            researcher_name=researcher,
            location=location,
            recording_date=recording_date,
            sequence_number=(
                int(sequence_number) if sequence_number and int(sequence_number) >= 1 else None
            ),
            sequence_label=sequence_label,
            event_context=event_context,
            source_gdrive_path=gdrive_path or None,
        )
        metadata.extraction_confidence = self._compute_confidence(metadata)
        return metadata

    def _extract_from_filename(self, name: str) -> dict[str, str | None]:
        """Extract metadata from the filename.

        Args:
            name: Original filename with extension.

        Returns:
            Dict with extracted fields (researcher, participant, date, etc.).
        """
        result: dict[str, str | None] = {}

        if not name:
            return result

        # Strip extension
        stem = PurePosixPath(name).stem

        # Strip device prefix like [Tablet]
        device_match = _DEVICE_PREFIX.match(stem)
        if device_match:
            result["device"] = device_match.group(1)
            stem = stem[device_match.end() :]

        # Camera files have no useful metadata in the name
        if _CAMERA_FILE.match(name):
            return result

        # Android VID files: extract date from timestamp
        vid_match = _VID_TIMESTAMP.match(name)
        if vid_match:
            year, month, day = vid_match.group(1), vid_match.group(2), vid_match.group(3)
            result["date"] = f"{day}-{month}-{year}"
            # Extract trailing sequence (e.g., ~3)
            tilde_seq = re.search(r"~(\d+)", stem)
            if tilde_seq:
                result["sequence_number"] = tilde_seq.group(1)
            return result

        # Extract sequence label (Parte I, etc.) before further parsing
        parte_match = _PARTE_LABEL.search(stem)
        if parte_match:
            result["sequence_label"] = parte_match.group(1)
            stem = stem[: parte_match.start()].rstrip()

        # Extract trailing sequence number
        seq_match = _TRAILING_SEQ.search(stem)
        if seq_match:
            result["sequence_number"] = seq_match.group(1)
            stem = stem[: seq_match.start()].rstrip(" -_")

        # Extract date from stem
        date = self._extract_date(stem)
        if date:
            result["date"] = date
            # Remove date from stem for name parsing
            stem = stem.replace(date, "").strip(" -_")

        # Now try to extract researcher/participant from remaining stem
        # Strategy 1: Underscore-delimited (Glenio_D.Elaine_30-07-2025_BARRA_20)
        if "_" in stem:
            self._parse_underscore_delimited(stem, result)
        else:
            # Strategy 2: Dash/space-delimited
            self._parse_dash_delimited(stem, result)

        return result

    def _parse_underscore_delimited(self, stem: str, result: dict[str, str | None]) -> None:
        """Parse underscore-delimited filenames.

        Handles patterns like:
        - Glenio_BarraDePelotas_30-07-2025_14
        - Dani_Henrique_15-11-2025_BARRA_08
        - Glenio_D.Elaine_30-07-2025_BARRA_20
        - DaniBorges_D.Maria_30-07-25_03

        Args:
            stem: Filename stem with dates and sequences already removed.
            result: Dict to populate with extracted fields.
        """
        parts = [p.strip() for p in stem.split("_") if p.strip()]

        # Remove parts that are pure dates or sequence numbers (already extracted)
        name_parts: list[str] = []
        for part in parts:
            if _DATE_DASHED.fullmatch(part):
                if not result.get("date"):
                    result["date"] = part
                continue
            if part.isdigit():
                if not result.get("sequence_number"):
                    result["sequence_number"] = part
                continue
            name_parts.append(part)

        if not name_parts:
            return

        # Check each part against known researchers and locations
        researchers_found: list[str] = []
        locations_found: list[str] = []
        remaining: list[str] = []

        for part in name_parts:
            matched_researcher = self._match_researcher(part)
            if matched_researcher:
                researchers_found.append(matched_researcher)
                continue
            matched_location = self._match_location(part)
            if matched_location:
                locations_found.append(matched_location)
                continue
            remaining.append(part)

        if researchers_found and not result.get("researcher"):
            result["researcher"] = researchers_found[0]
        if locations_found and not result.get("location"):
            result["location"] = locations_found[0]

        # Remaining parts are likely participant names
        if remaining and not result.get("participant"):
            result["participant"] = " ".join(remaining)

        # If only one name found and no researcher, first part is likely researcher
        if not result.get("researcher") and not result.get("participant") and name_parts:
            result["researcher"] = name_parts[0]
            if len(name_parts) > 1:
                result["participant"] = " ".join(name_parts[1:])

    def _parse_dash_delimited(self, stem: str, result: dict[str, str | None]) -> None:
        """Parse dash/space-delimited filenames.

        Handles patterns like:
        - Dani Borges-Pescador Henrique
        - Barra - Célia
        - Dani Borges 18-07-25
        - Dona Gilda 6- Dani Borges- 15 de out
        - Célia- Aúdio sobre fotos perdidas Dani Borges

        Args:
            stem: Filename stem with dates and sequences already removed.
            result: Dict to populate with extracted fields.
        """
        # Split on dash with surrounding spaces
        parts = [p.strip() for p in re.split(r"\s*-\s*", stem) if p.strip()]

        if not parts:
            return

        # Filter out numeric parts, date fragments, and descriptive phrases
        name_candidates: list[str] = []
        for part in parts:
            # Skip numbers (sequence numbers already extracted)
            if part.isdigit():
                continue
            # Skip date-like fragments (e.g., "12.06", "20 de jun.")
            cleaned = part.strip(" ,.\u200b")
            if _DATE_DOTTED.fullmatch(cleaned) or _DATE_WRITTEN.fullmatch(cleaned):
                continue
            if _DATE_DASHED.fullmatch(cleaned):
                continue
            # Skip long descriptive phrases (> 4 words likely a description, not a name)
            if len(part.split()) > 4:
                # But still check if it contains a known researcher
                for word_combo in self._iter_name_candidates(part):
                    matched = self._match_researcher(word_combo)
                    if matched and not result.get("researcher"):
                        result["researcher"] = matched
                continue
            name_candidates.append(part)

        if not name_candidates:
            return

        # Try to identify researcher vs participant
        for i, candidate in enumerate(name_candidates):
            matched_researcher = self._match_researcher(candidate)
            if matched_researcher:
                if not result.get("researcher"):
                    result["researcher"] = matched_researcher
                # Remaining candidates are participants
                others = [c for j, c in enumerate(name_candidates) if j != i]
                if others and not result.get("participant"):
                    result["participant"] = others[0]
                return

            matched_location = self._match_location(candidate)
            if matched_location:
                if not result.get("location"):
                    result["location"] = matched_location
                continue

        # No researcher match found — first candidate is likely the main name
        unmatched = [c for c in name_candidates if not self._match_location(c)]
        if len(unmatched) >= 2:
            # Heuristic: first part is researcher, second is participant
            result.setdefault("researcher", unmatched[0])
            result.setdefault("participant", unmatched[1])
        elif len(unmatched) == 1:
            result.setdefault("participant", unmatched[0])

    def _extract_from_path(self, gdrive_path: str) -> dict[str, str | None]:
        """Extract metadata from the Google Drive path.

        Args:
            gdrive_path: Full Google Drive path.

        Returns:
            Dict with extracted fields (location, event_context, etc.).
        """
        result: dict[str, str | None] = {}
        if not gdrive_path:
            return result

        parts = gdrive_path.split("/")

        # Find the location folder (after "IMAGENS e ÁUDIOS")
        location_folder = self._extract_location_folder(parts)
        if location_folder:
            folder_name = _LOCATION_SUFFIX.sub("", location_folder).strip()
            matched = self._match_location(folder_name)
            if matched:
                result["location"] = matched
            else:
                # Non-location folder (e.g., "Registros em vídeo- Glênio Rissio")
                # Try to extract researcher name from it
                for word in folder_name.replace("-", " ").split():
                    researcher = self._match_researcher(word)
                    if researcher:
                        result["researcher"] = researcher
                        break

        # Extract event context from subfolder
        event_context = self._extract_event_context(parts)
        if event_context:
            result["event_context"] = event_context
            # Try to extract participant and date from event context
            self._parse_event_context(event_context, result)

        return result

    def _extract_location_folder(self, path_parts: list[str]) -> str | None:
        """Find the location folder name from path segments.

        Args:
            path_parts: Path split by '/'.

        Returns:
            Location folder name or None.
        """
        for i, part in enumerate(path_parts):
            if _MEDIA_ROOT in part and i + 1 < len(path_parts):
                return path_parts[i + 1]
        return None

    def _extract_event_context(self, path_parts: list[str]) -> str | None:
        """Extract event context from subfolder names.

        Looks for subfolders like "Entrevista D. Silvia 29-11-2025 Barra de Pelotas",
        "Saída de campo Balsa 22-10-2025", or "Dona Gilda 12-11-25- Dani Borges".

        Args:
            path_parts: Path split by '/'.

        Returns:
            Event context string or None.
        """
        for i, part in enumerate(path_parts):
            if _MEDIA_ROOT in part:
                # Skip the location folder and media type folder (AUDIOS, VÍDEOS)
                # Look at deeper subfolders
                remaining = path_parts[i + 2 :]  # Skip location + media type
                for subfolder in remaining:
                    # Skip the filename itself (has a file extension) and generic folders
                    if re.search(r"\.\w{2,4}$", subfolder):
                        continue  # Likely a filename
                    if subfolder.upper() in ("AUDIOS", "AUDIOS ", "VÍDEOS", "FOTOS"):
                        continue
                    if subfolder.strip():
                        return subfolder.strip()
        return None

    def _parse_event_context(self, context: str, result: dict[str, str | None]) -> None:
        """Extract participant, researcher, and date from event context string.

        Handles patterns like:
        - "Entrevista D. Silvia 29-11-2025 Barra de Pelotas"
        - "[Glenio] Vídeos campo 30-07-2025 Barra de Pelotas"
        - "Dona Gilda 12-11-25- Dani Borges ORGANIZAR"

        Args:
            context: Event context subfolder name.
            result: Dict to populate.
        """
        # Extract bracketed researcher: [Glenio]
        bracket_match = re.match(r"\[([^\]]+)\]", context)
        if bracket_match:
            matched = self._match_researcher(bracket_match.group(1))
            if matched:
                result.setdefault("researcher", matched)

        # Extract date from context
        date = self._extract_date(context)
        if date:
            result.setdefault("date", date)

        # Look for "Entrevista" pattern to extract participant
        entrevista_match = re.match(r"Entrevista\s+(.+?)(?:\s+\d{2}-\d{2})", context, re.IGNORECASE)
        if entrevista_match:
            result.setdefault("participant", entrevista_match.group(1).strip())

    def _extract_date(self, text: str) -> str | None:
        """Extract the first date found in text.

        Args:
            text: Text to search for dates.

        Returns:
            Date string in original format, or None.
        """
        # Try DD-MM-YY(YY) first (most common)
        match = _DATE_DASHED.search(text)
        if match:
            return match.group(1)

        # Try written dates: "15 de out"
        match = _DATE_WRITTEN.search(text)
        if match:
            return match.group(1)

        # Try dotted dates: "20.05"
        match = _DATE_DOTTED.search(text)
        if match:
            return match.group(1)

        return None

    def _match_researcher(self, text: str) -> str | None:
        """Check if text matches a known researcher name.

        Args:
            text: Text to match.

        Returns:
            Canonical researcher name or None.
        """
        normalized = text.strip().lower()
        # Direct match
        if normalized in self._researchers:
            return self._researchers[normalized]
        # Partial match: check if any researcher name starts with the text
        for key, canonical in self._researchers.items():
            if key.startswith(normalized) and len(normalized) >= 4:
                return canonical
            if normalized.startswith(key):
                return canonical
        return None

    def _match_location(self, text: str) -> str | None:
        """Check if text matches a known location.

        Args:
            text: Text to match.

        Returns:
            Canonical location name or None.
        """
        normalized = text.strip().lower()
        # Also try without accents for matching like "BarraDePelotas"
        collapsed = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).lower()

        # Exact matches first
        for key, canonical in self._locations.items():
            if key in (normalized, collapsed):
                return canonical

        # Substring matches, longest key first to prefer specific over generic
        sorted_locs = sorted(self._locations.items(), key=lambda x: len(x[0]), reverse=True)
        for key, canonical in sorted_locs:
            if key in normalized or key in collapsed:
                return canonical
        return None

    def _iter_name_candidates(self, text: str) -> list[str]:
        """Generate name candidates from a longer text phrase.

        Args:
            text: Text that might contain a name embedded in a phrase.

        Returns:
            List of 1-2 word combinations to test.
        """
        words = text.split()
        candidates: list[str] = []
        for i in range(len(words)):
            candidates.append(words[i])
            if i + 1 < len(words):
                candidates.append(f"{words[i]} {words[i + 1]}")
        return candidates

    @staticmethod
    def _sanitize_name(name: str | None) -> str | None:
        """Validate and clean a person name.

        Strips trailing numbers (e.g., "Dona Gilda 1" -> "Dona Gilda")
        and rejects candidates that are purely numeric or look like
        descriptive phrases rather than proper names. In Portuguese,
        person names are title-cased (each word capitalized), while
        descriptive phrases have lowercase words (e.g., "Problemas urbanos").

        Args:
            name: Candidate name string.

        Returns:
            The cleaned name if valid, None otherwise.
        """
        if not name:
            return None
        stripped = name.strip(" -_,.")
        if not stripped:
            return None
        # Strip trailing numbers (sequence artifacts like "Dona Gilda 1")
        stripped = re.sub(r"\s+\d+$", "", stripped).rstrip(" -_,.")
        # Reject if still contains digits (e.g., "2025VID", "12.06")
        if not stripped or re.search(r"\d", stripped):
            return None
        # Multi-word names must be title-cased (allow honorifics like "D.")
        words = stripped.split()
        if len(words) > 1 and any(
            w[0].islower() for w in words if len(w) > 2 and not w.startswith("D.")
        ):
            return None
        return stripped

    def _compute_confidence(self, metadata: SourceMetadata) -> float:
        """Compute extraction confidence as fraction of core fields extracted.

        Args:
            metadata: The extracted metadata.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        core_fields = [
            metadata.participant_name,
            metadata.researcher_name,
            metadata.location,
            metadata.recording_date,
        ]
        filled = sum(1 for f in core_fields if f is not None)
        return round(filled / len(core_fields), 2)
