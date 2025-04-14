# Leitfaden zur Agenten-basierten Suche

Dieser Leitfaden erklärt die agenten-basierte Suche im Spiegel RAG-System im Detail und gibt Hinweise zur effektiven Nutzung.

## Überblick

Die agenten-basierte Suche ist ein fortschrittlicher Hybrid-Ansatz, der Retrieval und Analyse in einem mehrstufigen Prozess kombiniert. Dabei werden zunächst mehr Texte abgerufen als bei der Standard-Suche, und dann vom LLM basierend auf ihrer Relevanz für die gestellte Frage bewertet und gefiltert.

## Wie funktioniert es?

Der Prozess läuft in mehreren Stufen ab:

1. **Initiale Retrieval-Phase**: Das System ruft eine größere Menge an Texten ab (z.B. 100 statt der üblichen 10)
2. **Bewertungsphase**: Das LLM bewertet jeden Text hinsichtlich seiner Relevanz für die gestellte Frage
3. **Mehrstufige Filterung**: In mehreren Stufen wird die Menge der Texte reduziert (z.B. 100 → 50 → 20 → 10)
4. **Antwortgenerierung**: Das LLM generiert eine Antwort basierend auf den bestbewerteten Texten

## Unterschied zwischen Inhaltsbeschreibung und Frage

Ein wichtiges Konzept in der agenten-basierten Suche ist die Unterscheidung zwischen Inhaltsbeschreibung und Frage:

### Inhaltsbeschreibung

- Bestimmt, **welche Art von Inhalten** gesucht werden sollen
- Wird für die initiale Vektorsuche verwendet
- Kann breiter gefasst sein als die eigentliche Frage
- Beispiel: "Artikel über politische Skandale in Deutschland"

### Frage

- Bestimmt, **welche Information** gesucht wird
- Wird für die Bewertung der Relevanz der gefundenen Texte verwendet
- Ist typischerweise spezifischer als die Inhaltsbeschreibung
- Beispiel: "Wie unterschied sich die Berichterstattung über Skandale verschiedener politischer Parteien?"

Wenn keine separate Inhaltsbeschreibung angegeben wird, wird die Frage für beide Zwecke verwendet.

## Anwendungsszenarien

### 1. Analytische Fragen

**Beispiel:**

- **Frage**: "Wie veränderte sich die Darstellung der DDR in den 1970er Jahren im Vergleich zu den 1950er Jahren?"
- **Inhaltsbeschreibung**: "Berichterstattung über die DDR"

Hier hilft der Agent, Texte zu identifizieren, die tatsächlich vergleichende Elemente enthalten.

### 2. Thematische Muster

**Beispiel:**

- **Frage**: "Welche wiederkehrenden Narrative werden in der Berichterstattung über die Sowjetunion verwendet?"
- **Inhaltsbeschreibung**: "Artikel über die Sowjetunion und ihre Politik"

Der Agent kann hier Texte identifizieren, die typische Sprachmuster und Erzählstrukturen aufweisen.

### 3. Indirekte Zusammenhänge

**Beispiel:**

- **Frage**: "Wie wird der Zusammenhang zwischen Wirtschaftspolitik und sozialen Bewegungen dargestellt?"
- **Inhaltsbeschreibung**: "Artikel über Wirtschaftspolitik oder soziale Bewegungen"

Der Agent kann Texte finden, die diese Verbindung herstellen, auch wenn sie nicht explizit beide Themen im Fokus haben.

## Empfehlungen für die Konfiguration

### Filtereinstellungen

Die Filtereinstellungen bestimmen, wie viele Texte in jeder Phase des Prozesses behalten werden:

- **Initiale Textmenge**:

  - Für breite Themen: 100-200 Texte
  - Für spezifische Themen: 50-100 Texte
- **Filterstufen**:

  - Empfohlene Abstufung: Etwa Halbierung in jeder Stufe
  - Beispiel: 100 → 50 → 25 → 10
  - Mindestens 3 Stufen für optimale Filterung

### Modellauswahl

- Für komplexe Bewertungen wird **GPT-4o** empfohlen
- **HU-LLM** ist für einfachere Bewertungen ausreichend

### System-Prompt

Der Standard-System-Prompt ist für die meisten Anwendungsfälle geeignet. Spezielle Anpassungen können sinnvoll sein für:

- Medienkritische Analysen: Wählen Sie die "media_critique" Vorlage
- Historische Kontextualisierung: Wählen Sie die "historical_analysis" Vorlage

## Interpretation der Ergebnisse

Nach Abschluss der Suche werden angezeigt:

1. **Filterungsprozess**: Visualisierung der einzelnen Filterungsstufen
2. **Textbewertungen**: Erklärungen, warum bestimmte Texte ausgewählt wurden
3. **Gefundene Texte**: Die tatsächlichen Textinhalte der ausgewählten Dokumente
4. **Metadaten**: Informationen zum Suchprozess

Besonders die Textbewertungen sind wertvoll, da sie Einblick geben, warum das LLM bestimmte Texte als relevant erachtet hat.

## Tipps für effektive Suchen

1. **Unterschiedliche Inhaltsbeschreibung und Frage verwenden**

   - Inhaltsbeschreibung: Breiter fassen, um mehr potenzielle Texte zu erfassen
   - Frage: Präzise formulieren, um die Bewertung zu fokussieren
2. **Analytische Fragen stellen**

   - Fragen nach Mustern, Vergleichen, Entwicklungen und Zusammenhängen
   - "Wie", "Warum", "Inwiefern" als Frageeinleitungen
3. **Mit den Filterstufen experimentieren**

   - Bei unbefriedigenden Ergebnissen: Mehr initiale Texte abrufen
   - Bei zu allgemeinen Ergebnissen: Strengere Filterung in den späteren Stufen
4. **Die Bewertungen analysieren**

   - Achten Sie auf wiederkehrende Begründungen in den Textbewertungen
   - Diese können Hinweise auf dominante Themen oder Narrative geben
5. **Kombinieren mit Standard-RAG**

   - Verwenden Sie die agenten-basierte Suche für explorative Analysen
   - Nutzen Sie die Standard-Suche für gezielte Nachfragen zu gefundenen Aspekten

## Einschränkungen

- **Zeitaufwand**: Die agenten-basierte Suche dauert deutlich länger als die Standard-Suche
- **LLM-Limitierungen**: Die Bewertungen können subjektiv sein und vom verwendeten Modell abhängen
- **Kontextlimitierungen**: Sehr lange Texte können vom LLM nicht vollständig erfasst werden
- **Ressourcenbedarf**: Höherer Rechenaufwand und API-Kosten (bei Verwendung von OpenAI)

## Fallbeispiele

### Beispiel 1: Analyse politischer Berichterstattung

**Frage**: "Wie unterscheidet sich die Darstellung von SPD- und CDU-Politikern in Krisenzeiten?"
**Inhaltsbeschreibung**: "Artikel über SPD- oder CDU-Politiker in politischen oder wirtschaftlichen Krisen"

**Ergebnis**: Der Agent findet Texte, die beide Parteien erwähnen und direkte Vergleiche ermöglichen, anstatt nur Texte über einzelne Parteien.

### Beispiel 2: Wirtschaftsberichterstattung im Wandel

**Frage**: "Wie veränderte sich die Bewertung amerikanischer Wirtschaftspolitik zwischen den 1950er und 1970er Jahren?"
**Inhaltsbeschreibung**: "Berichterstattung über amerikanische Wirtschaft und Wirtschaftspolitik"

**Ergebnis**: Der Agent identifiziert Texte, die evaluative Elemente enthalten und Wertungen über verschiedene Zeiträume hinweg ermöglichen.
