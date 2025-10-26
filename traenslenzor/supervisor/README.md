# Graph

```mermaid
flowchart TD
    IntentClassifier --> Abort
    Abort --> IntentClassifier
    IntentClassifier --> ImageLoader

    ImageLoader --> ImageVerifyer{ImageVerifyer}
    ImageVerifyer -->|Valid| RequestClarification{RequestClarification}
    ImageVerifyer -->|Invalid Interrupt| IntentClassifier

    IntentClassifier --> LanguageDetection
    LanguageDetection --> RequestClarification

    IntentClassifier --> OptimizationDetection
    OptimizationDetection --> RequestClarification

    RequestClarification -->|Incomplete Interrupt| IntentClassifier

    RequestClarification -->|Complete Interrupt| DetermineSatisfied
    DetermineSatisfied -->|Unsatisfied Interrupt| IntentClassifier
    DetermineSatisfied --> LayoutDetector
    DetermineSatisfied --> FontDetector
    LayoutDetector --> Translator
    Translator --> DocumentRenderer
    FontDetector --> DocumentRenderer
    DocumentRenderer --> IntentClassifier
```

