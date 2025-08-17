import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { GoogleGenAI, GenerateContentResponse, Type } from '@google/genai';
import { TALKERS_CAVE_SCENES, TALKERS_CAVE_SCENE_IMAGES, TALKERS_CAVE_CHARACTER_IMAGES, TALKERS_CAVE_SCENE_BACKGROUNDS } from '../constants';
import { MicrophoneIcon } from './icons/MicrophoneIcon';
import { PracticeSoundIcon } from './icons/PracticeSoundIcon';
import { guidelines as allGuidelines } from '../guidelines';

declare global {
  interface SpeechRecognition {
    lang: string;
    continuous: boolean;
    interimResults: boolean;
    maxAlternatives: number;
    start(): void;
    stop(): void;
    abort(): void;
    onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any) | null;
    onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => any) | null;
    onend: ((this: SpeechRecognition, ev: Event) => any) | null;
    onstart: ((this: SpeechRecognition, ev: Event) => any) | null;
  }

  interface Window {
    SpeechRecognition: { new(): SpeechRecognition };
    webkitSpeechRecognition: { new(): SpeechRecognition };
  }
  interface SpeechRecognitionEvent extends Event {
    readonly resultIndex: number;
    readonly results: SpeechRecognitionResultList;
  }
  interface SpeechRecognitionResultList {
    readonly length: number;
    item(index: number): SpeechRecognitionResult;
    [index: number]: SpeechRecognitionResult;
  }
  interface SpeechRecognitionResult {
    readonly isFinal: boolean;
    readonly length: number;
    item(index: number): SpeechRecognitionAlternative;
    [index: number]: SpeechRecognitionAlternative;
  }
  interface SpeechRecognitionAlternative {
    readonly transcript: string;
    readonly confidence: number;
  }
  interface SpeechRecognitionErrorEvent extends Event {
    readonly error: string;
    readonly message: string;
  }
}

// Type definitions
type Step = 'SCENE' | 'CHARACTER' | 'LOADING_SCRIPT' | 'GAME' | 'ANALYZING_PERFORMANCE' | 'PRACTICE_PREP' | 'PRACTICE' | 'COMPLETE';
type Scene = keyof typeof TALKERS_CAVE_SCENES;
type ScriptLine = { character: string; line: string };
type Mistake = { said: string; expected: string };
type PracticeWord = { word: string; phonemes: string[] };

// Props
interface TalkersCaveGameProps {
  onComplete: (success: boolean) => void;
  userGrade: number;
  currentLevel: number;
  onBackToGrades: () => void;
  language: string;
}

// Guideline types for type safety
interface LevelHint {
  level: number;
  target_word_count: number;
  sentence_types_hint: string[];
}

interface Grade {
  grade: number;
  reading_level_summary: string;
  word_range: { min: number; max: number };
  sentence_types_allowed: string[];
  syllables_per_word_target: { source_text: string };
  phonics_or_morphology_focus: string;
  sight_words: { examples: string[] };
  punctuation_allowed: string[];
  topic_suggestions: string[];
  level_progression_hint: LevelHint[];
}

interface Guidelines {
  global_rules: Record<string, any>;
  grades: Grade[];
}

// Utility Functions
const cleanWord = (word: string) => word.trim().toLowerCase().replace(/[.,?!]/g, '');

const extractJson = <T,>(text: string): T | null => {
    try {
        let jsonStr = text.trim();
        const fenceRegex = /```(?:json)?\s*([\s\S]*?)\s*```/;
        const fenceMatch = jsonStr.match(fenceRegex);

        if (fenceMatch && fenceMatch[1]) {
            jsonStr = fenceMatch[1].trim();
        } else {
            const firstBracket = jsonStr.indexOf('[');
            const lastBracket = jsonStr.lastIndexOf(']');
            const firstBrace = jsonStr.indexOf('{');
            const lastBrace = jsonStr.lastIndexOf('}');

            let start = -1;
            let end = -1;
            
            if (firstBracket !== -1 && firstBrace !== -1) start = Math.min(firstBracket, firstBrace);
            else if (firstBracket !== -1) start = firstBracket;
            else start = firstBrace;

            if (lastBracket !== -1 && lastBrace !== -1) end = Math.max(lastBracket, lastBrace);
            else if (lastBracket !== -1) end = lastBracket;
            else end = lastBrace;

            if (start !== -1 && end > start) {
                jsonStr = jsonStr.substring(start, end + 1);
            }
        }
        return JSON.parse(jsonStr) as T;
    } catch (e) {
        console.error("Failed to parse JSON from response:", text, e);
        return null;
    }
};

const getPhoneticBreakdown = async (word: string, language: string): Promise<string[]> => {
    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
        const languageName = language === 'hi' ? 'Hindi' : 'English';
        const systemInstruction = `You are a linguistic expert specializing in ${languageName} phonetics for children. Your task is to break down a given word into its distinct phonetic syllables. 
**RULES:**
- The syllables should be simple, intuitive, and easy for a child to read and pronounce. For example, for "together", respond with ["tu", "geh", "dhuh"].
- Your response MUST be a single, valid JSON array of strings.
- Do NOT use markdown code fences or any other text outside the JSON array.
- If the word is a single syllable, return an array with that one word.`;

        const prompt = `Break down the word "${word}" into phonetic syllables.`;
        const responseSchema = { type: Type.ARRAY, items: { type: Type.STRING } };

        const response: GenerateContentResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
            config: { systemInstruction, responseMimeType: 'application/json', responseSchema },
        });
        
        const result = extractJson<string[]>(response.text);
        return Array.isArray(result) && result.every(item => typeof item === 'string') ? result : [word];
    } catch (e) {
        console.error(`Phonetic breakdown failed for "${word}":`, e);
        return [word];
    }
};

const analyzeReadingWithAI = async (spokenText: string, targetText: string, language: string): Promise<string[]> => {
    if (!spokenText.trim()) return targetText.split(' ').filter(Boolean);

    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
        const languageName = language === 'hi' ? 'Hindi' : 'English';
        const systemInstruction = `You are a highly precise ${languageName} pronunciation analyst for a children's learning application. Your task is to compare a 'target text' with a 'spoken text' and return a JSON array of words from the target text that were either mispronounced or omitted.

**CRITICAL RULES FOR ANALYSIS:**
1.  **Exact Word Matching:** Your primary goal is to check if the child said the words from the target text. The comparison must be case-insensitive. Ignore all punctuation (like commas, periods, question marks).
2.  **Strict Omission and Mispronunciation Detection:**
    *   **Omission:** If a word from the target text is missing from the spoken text, it's a mistake.
    *   **Mispronunciation:** If a word is spoken but is phonetically very different from the target word, it's a mistake. Be strict. Do not be overly lenient. For example, if the target is "apple" and the user says "able", it is a mistake.
    *   **Output:** You only need to return the *target* word that was the source of the error.
3.  **Ignore Inserted Words:** You MUST ignore any extra words the child says that are not in the target text. For example, if the target is "I see a cat" and the child says "Um, I see a big cat", there are no mistakes. "big" is an insertion and should be ignored.
4.  **Perfect Match:** If the spoken text correctly includes all words from the target text (ignoring insertions), you MUST return an empty array \`[]\`.

**RESPONSE FORMAT:**
- Your response MUST be a single, valid JSON array of strings.
- Each string in the array must be a single, lowercase word from the target text that was identified as a mistake.
- Do NOT include markdown fences (\`\`\`json) or any other text. The response must start with \`[\` and end with \`]\`.

**EXAMPLES:**
*   **Example 1 (Omission):** Target Text: "The big red dog." / Spoken Text: "The big dog." -> Your Response: \`["red"]\`
*   **Example 2 (Mispronunciation):** Target Text: "She wants to play." / Spoken Text: "She wants to pray." -> Your Response: \`["play"]\`
*   **Example 3 (Insertion - No Mistakes):** Target Text: "Let's go home." / Spoken Text: "Um, let's go home now." -> Your Response: \`[]\`
*   **Example 4 (Multiple Errors):** Target Text: "My favorite animal is the turtle." / Spoken Text: "My favorite animal the table." -> Your Response: \`["is", "turtle"]\`
*   **Example 5 (Perfect Match):** Target Text: "Can I have another?" / Spoken Text: "Can I have another?" -> Your Response: \`[]\``;
        
        const prompt = `Analyze the spoken text against the target text.\n\nTarget Text: "${targetText}"\nSpoken Text: "${spokenText}"\n\nReturn the incorrect lowercase words as a JSON array of strings.`;
        const responseSchema = { type: Type.ARRAY, items: { type: Type.STRING } };

        const response: GenerateContentResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
            config: { systemInstruction, responseMimeType: 'application/json', responseSchema },
        });
        
        const result = extractJson<string[]>(response.text);
        return Array.isArray(result) && result.every(item => typeof item === 'string') ? result : [];
    } catch (e) {
        console.error("AI analysis failed:", e);
        return []; 
    }
};

export const TalkersCaveGame: React.FC<TalkersCaveGameProps> = ({ onComplete, userGrade, currentLevel, onBackToGrades, language }) => {
  const [step, setStep] = useState<Step>('SCENE');
  const [selectedScene, setSelectedScene] = useState<Scene | null>(null);
  const [centeredScene, setCenteredScene] = useState<Scene>('Shopkeeper and Customer');
  const [selectedCharacter, setSelectedCharacter] = useState<string | null>(null);
  const [script, setScript] = useState<ScriptLine[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [currentTurn, setCurrentTurn] = useState(0);
  const [isAiSpeaking, setIsAiSpeaking] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [isRecognitionActive, setIsRecognitionActive] = useState(false);
  const [recognitionError, setRecognitionError] = useState<string | null>(null);
  const [mistakes, setMistakes] = useState<Mistake[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [userDialog, setUserDialog] = useState<{ target: string; said: string }[]>([]);
  const [userTranscript, setUserTranscript] = useState('');

  const [practiceWords, setPracticeWords] = useState<PracticeWord[]>([]);
  const [currentPracticeWordIndex, setCurrentPracticeWordIndex] = useState(0);
  const [practiceStatus, setPracticeStatus] = useState<'IDLE' | 'LISTENING' | 'SUCCESS' | 'TRY_AGAIN'>('IDLE');
  const [practiceTranscript, setPracticeTranscript] = useState('');
  
  const hasProcessedTurn = useRef(false);
  const wasStoppedIntentionally = useRef(false);
  
  // Refs for practice recognizer
  const practiceRecognizer = useRef<SpeechRecognition | null>(null);
  const practiceAttemptResult = useRef<'SUCCESS' | 'TRY_AGAIN' | 'NO_RESULT' | null>(null);
  const practiceResultHandlerRef = useRef((_event: SpeechRecognitionEvent) => {});
  const practiceEndHandlerRef = useRef((_event: Event) => {});
  const practiceErrorHandlerRef = useRef((_event: SpeechRecognitionErrorEvent) => {});

  const processUserTurn = useCallback((transcript: string) => {
    if (hasProcessedTurn.current) return;
    hasProcessedTurn.current = true;

    const targetLine = script[currentTurn].line;
    setUserDialog(prev => [...prev, { target: targetLine, said: transcript }]);
    
    if (currentTurn < script.length - 1) {
      setCurrentTurn(prev => prev + 1);
    } else {
      setStep('ANALYZING_PERFORMANCE');
    }
  }, [script, currentTurn]);

  const processUserTurnRef = useRef(processUserTurn);
  useEffect(() => { processUserTurnRef.current = processUserTurn; }, [processUserTurn]);

  // Effect to set up practice recognizer ONCE
  useEffect(() => {
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionAPI) {
        setRecognitionError("Speech recognition not supported in this browser.");
        return;
    };
    
    const recognizer = new SpeechRecognitionAPI();
    practiceRecognizer.current = recognizer;
    recognizer.lang = language === 'hi' ? 'hi-IN' : 'en-IN';
    recognizer.continuous = false;
    recognizer.interimResults = false;

    recognizer.onresult = (event) => practiceResultHandlerRef.current(event);
    recognizer.onerror = (event) => practiceErrorHandlerRef.current(event);
    recognizer.onend = (event) => practiceEndHandlerRef.current(event);

    return () => {
        recognizer.onresult = null;
        recognizer.onerror = null;
        recognizer.onend = null;
        if(recognizer.abort) recognizer.abort();
    };
  }, [language]);

  // Effect to update practice recognizer HANDLER LOGIC when state changes
  useEffect(() => {
    practiceResultHandlerRef.current = (event: SpeechRecognitionEvent) => {
        const transcript = event.results?.[0]?.[0]?.transcript.trim();
        if (!transcript) {
          practiceAttemptResult.current = 'NO_RESULT';
          return;
        }

        setPracticeTranscript(transcript);

        if (practiceWords.length > 0 && currentPracticeWordIndex < practiceWords.length) {
            const targetWord = practiceWords[currentPracticeWordIndex].word;
            if (cleanWord(transcript) === cleanWord(targetWord)) {
                practiceAttemptResult.current = 'SUCCESS';
            } else {
                practiceAttemptResult.current = 'TRY_AGAIN';
            }
        }
    };

    practiceErrorHandlerRef.current = (event: SpeechRecognitionErrorEvent) => {
      if (event.error !== 'no-speech' && event.error !== 'aborted') {
        setRecognitionError(`Mic error: "${event.error}".`);
      }
      practiceAttemptResult.current = 'NO_RESULT';
    };

    practiceEndHandlerRef.current = (_event: Event) => {
      if (practiceAttemptResult.current === 'SUCCESS') {
        setPracticeStatus('SUCCESS');
        setTimeout(() => {
          if (currentPracticeWordIndex < practiceWords.length - 1) {
            setCurrentPracticeWordIndex(prev => prev + 1);
            setPracticeStatus('IDLE');
            setPracticeTranscript('');
          } else {
            onComplete(true);
          }
        }, 1500);
      } else if (practiceAttemptResult.current === 'TRY_AGAIN') {
        setPracticeStatus('TRY_AGAIN');
      } else {
        setPracticeStatus('IDLE');
      }
      practiceAttemptResult.current = null;
    };
  }, [practiceWords, currentPracticeWordIndex, onComplete]);


  useEffect(() => {
    if (!window.speechSynthesis) return;
    const loadVoices = () => setVoices(window.speechSynthesis.getVoices());
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
    return () => { window.speechSynthesis.onvoiceschanged = null; };
  }, []);

  const speechRecognizer = useMemo<SpeechRecognition | null>(() => {
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionAPI) return null;
    const recognizer = new SpeechRecognitionAPI();
    recognizer.lang = language === 'hi' ? 'hi-IN' : 'en-IN';
    recognizer.maxAlternatives = 1;
    return recognizer;
  }, [language]);
  
  useEffect(() => {
    if (!speechRecognizer) setRecognitionError('Speech recognition is not supported in this browser. Try Chrome or Edge.');
  }, [speechRecognizer]);

  useEffect(() => {
    if (!speechRecognizer || step !== 'GAME') return;

    const transcriptRef = { current: '' };

    speechRecognizer.continuous = true;
    speechRecognizer.interimResults = true;

    const handleStart = () => {
        transcriptRef.current = '';
        setUserTranscript('');
        setIsRecognitionActive(true);
    };

    const handleResult = (event: SpeechRecognitionEvent) => {
        if (hasProcessedTurn.current) return;
        const fullTranscript = Array.from(event.results)
            .map(result => result[0])
            .map(result => result.transcript)
            .join('');
        
        transcriptRef.current = fullTranscript;
        setUserTranscript(fullTranscript);
    };

    const handleEnd = () => {
        setIsRecognitionActive(false);
        if (wasStoppedIntentionally.current && !hasProcessedTurn.current) {
            processUserTurnRef.current(transcriptRef.current.trim());
        }
        wasStoppedIntentionally.current = false;
    };

    const handleError = (event: SpeechRecognitionErrorEvent) => {
        const errorType = event.error;
        if (['aborted', 'no-speech'].includes(errorType)) {
            return;
        }
        if (errorType === 'not-allowed' || errorType === 'service-not-allowed') {
            setRecognitionError('Microphone access denied.');
        } else {
            setRecognitionError(`Mic error: "${errorType}".`);
        }
    };
    
    speechRecognizer.onstart = handleStart;
    speechRecognizer.onresult = handleResult;
    speechRecognizer.onend = handleEnd;
    speechRecognizer.onerror = handleError;
    
    return () => {
      speechRecognizer.onresult = null;
      speechRecognizer.onstart = null;
      speechRecognizer.onend = null;
      speechRecognizer.onerror = null;
    };
  }, [speechRecognizer, step]);
  
  const speak = useCallback((text: string, onEndCallback: () => void) => {
    if (!window.speechSynthesis || voices.length === 0) {
      setTimeout(onEndCallback, text.length * 50);
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voiceLang = language === 'hi' ? 'hi-IN' : 'en-IN';

    const preferredVoice = voices.find(v => v.lang === voiceLang && v.name.includes('Google')) || voices.find(v => v.lang === voiceLang);
    
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    } else {
      const fallbackVoice = voices.find(v => v.lang.startsWith(language)) || voices.find(v => v.lang.startsWith('en'));
      if (fallbackVoice) utterance.voice = fallbackVoice;
    }

    utterance.onstart = () => setIsAiSpeaking(true);
    utterance.onend = () => { setIsAiSpeaking(false); onEndCallback(); };
    utterance.onerror = () => { setIsAiSpeaking(false); onEndCallback(); };
    window.speechSynthesis.speak(utterance);
  }, [voices, language]);

  useEffect(() => {
    if (step !== 'ANALYZING_PERFORMANCE') return;
    
    const analyzeAllDialog = async () => {
        setIsAnalyzing(true);
        const allIncorrectWords: string[] = [];
        for (const dialog of userDialog) {
            const turnIncorrectWords = await analyzeReadingWithAI(dialog.said, dialog.target, language);
            allIncorrectWords.push(...turnIncorrectWords);
        }
        
        const allMistakes: Mistake[] = [...new Set(allIncorrectWords)].map(word => ({ expected: word, said: '' }));
        setMistakes(allMistakes);
        setIsAnalyzing(false);
        
        if (allMistakes.length > 0) {
            setStep('PRACTICE_PREP');
        } else {
            setStep('COMPLETE');
        }
    };

    analyzeAllDialog();
  }, [step, userDialog, language]);

  useEffect(() => {
    if (step !== 'PRACTICE_PREP') return;
    const preparePractice = async () => {
        const wordsToPractice = [...new Set(mistakes.map(m => m.expected).filter(Boolean))];
        if (wordsToPractice.length === 0) { onComplete(true); return; }
        
        const phoneticData: PracticeWord[] = [];
        for (const word of wordsToPractice) {
            const phonemes = await getPhoneticBreakdown(word, language);
            phoneticData.push({ word, phonemes });
        }

        setPracticeWords(phoneticData.filter(p => p.phonemes.length > 0));
        setCurrentPracticeWordIndex(0);
        setPracticeStatus('IDLE');
        setPracticeTranscript('');
        setStep('PRACTICE');
    };
    preparePractice();
  }, [step, mistakes, onComplete, language]);

  const handlePracticeMicClick = useCallback(() => {
    const recognizer = practiceRecognizer.current;
    if (!recognizer || practiceStatus === 'LISTENING' || practiceStatus === 'SUCCESS') {
        return;
    }
    
    if (practiceStatus === 'IDLE' || practiceStatus === 'TRY_AGAIN') {
        try {
            practiceAttemptResult.current = null;
            setPracticeTranscript('');
            setPracticeStatus('LISTENING');
            recognizer.start();
        } catch (e) {
            console.error("Could not start practice recognition:", e);
            setRecognitionError("Mic failed to start. Please try again.");
            setPracticeStatus('IDLE');
        }
    }
  }, [practiceStatus]);

  const generateScript = useCallback(async (scene: Scene, character: string) => {
    setError(null);
    setStep('LOADING_SCRIPT');
    
    try {
        const loadedGuidelines = allGuidelines as Guidelines;

        const aiCharacter = TALKERS_CAVE_SCENES[scene].find(c => c !== character);
        if (!aiCharacter) throw new Error("Could not determine AI character.");

        const gradeData = loadedGuidelines.grades.find((g: any) => g.grade === userGrade);
        if (!gradeData) {
            throw new Error(`Guidelines for grade ${userGrade} not found in orf_content_guidelines.json.`);
        }
        
        const internalLevelIndex = (currentLevel - 1) % gradeData.level_progression_hint.length;
        const levelHint = gradeData.level_progression_hint[internalLevelIndex];
        const { global_rules } = loadedGuidelines;
        const sightWordExamples = gradeData.sight_words.examples.filter((_:any, i: number) => i % 2 === 0).join(', ');

        const languageName = language === 'hi' ? 'Hindi' : 'English';
        const systemInstruction = `You are a scriptwriter for a children's English learning game. Your task is to generate a natural, engaging, and age-appropriate conversation script based on extremely specific educational guidelines.

**SCRIPT LANGUAGE:** The entire script, including all lines for all characters, MUST be in ${languageName}.

**GLOBAL RULES (Non-negotiable):**
- **Context:** ${global_rules.context}
- **Tone:** ${global_rules.tone}
- **Language Control:** ${global_rules.language_control}
- **Text Hygiene:** ${global_rules.text_hygiene}
- **Prohibited Content:** Do not include content with disallowed themes (violence, explicit politics, prejudice), heavy jargon, typos, or formatting issues.

**GRADE ${userGrade}, LEVEL ${levelHint.level} SPECIFIC GUIDELINES:**
- **Reading Level Summary:** ${gradeData.reading_level_summary}
- **Word Count:** The total word count for all 4 of the user's lines COMBINED should be between ${gradeData.word_range.min} and ${gradeData.word_range.max} words. Aim for the target of ${levelHint.target_word_count} words for the user's total lines.
- **Sentence Types:** Use only these sentence types: ${gradeData.sentence_types_allowed.join(', ')}. The complexity should match these hints: ${levelHint.sentence_types_hint.join(', ')}.
- **Syllable Complexity:** ${gradeData.syllables_per_word_target.source_text}
- **Phonics/Morphology Focus:** ${gradeData.phonics_or_morphology_focus}
- **Punctuation:** Only use the following: ${gradeData.punctuation_allowed.join(', ')}.
- **Topics:** Draw inspiration from these topics: ${gradeData.topic_suggestions.join(', ')}.
- **Sight Words:** If it fits naturally, include some of these words: ${sightWordExamples}.

**SCRIPT REQUIREMENTS:**
- The scene is: "${scene}".
- The user plays as "${character}". The AI plays as "${aiCharacter}".
- The AI character ("${aiCharacter}") MUST speak first.
- The script must be exactly 8 turns (lines) long in total.
- Your entire response MUST be a single, valid JSON array of 8 objects.
- Each object must have "character" and "line" string properties.
- Do NOT use markdown code fences or any other text outside the JSON response.`;
        
        const prompt = `Generate the 8-line script for Grade ${userGrade} (Level ${levelHint.level}) in ${languageName} now.`;

        const scriptSchema = {
            type: Type.ARRAY,
            items: {
                type: Type.OBJECT,
                properties: { character: { type: Type.STRING }, line: { type: Type.STRING } },
                required: ['character', 'line']
            }
        };
        
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
        const response: GenerateContentResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
            config: { systemInstruction, responseMimeType: "application/json", responseSchema: scriptSchema }
        });

        const parsedScript = extractJson<ScriptLine[]>(response.text);

        if (Array.isArray(parsedScript) && parsedScript.length === 8 && parsedScript.every(item => 'character' in item && 'line' in item)) {
            setScript(parsedScript);
            setStep('GAME');
            setCurrentTurn(0);
        } else {
            throw new Error('Received invalid script format from API. Expected a JSON array of 8 script lines.');
        }
    } catch (e) {
        console.error(e);
        const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred.';
        setError(`Sorry, I couldn't create a script. ${errorMessage} Please try again.`);
        setStep('CHARACTER');
    }
  }, [userGrade, currentLevel, language]);

  const startRecognition = useCallback(() => {
    if (!speechRecognizer || isRecognitionActive) return;
    try {
        hasProcessedTurn.current = false;
        wasStoppedIntentionally.current = false;
        setRecognitionError(null);
        speechRecognizer.start();
    } catch(e: any) {
        if (e.name !== 'InvalidStateError') { console.error("Could not start recognition:", e); setRecognitionError("Failed to start microphone."); }
    }
  }, [speechRecognizer, isRecognitionActive]);

  const handleMicButtonClick = () => {
    if (isRecognitionActive) {
      wasStoppedIntentionally.current = true;
      if (speechRecognizer) {
        speechRecognizer.stop();
      }
    } else {
      startRecognition();
    }
  };

  useEffect(() => {
    if (step !== 'GAME' || !script.length || currentTurn >= script.length || recognitionError || isAiSpeaking) return;
    const currentLine = script[currentTurn];
    const isUserTurn = currentLine.character === selectedCharacter;
    
    if (isUserTurn) {
      // User will now manually click the microphone button
    } else {
        if (speechRecognizer) try { speechRecognizer.stop(); } catch (e) {}
        const handleAiTurnEnd = () => {
            if (currentTurn < script.length - 1) {
              setCurrentTurn(prev => prev + 1);
            } else {
              setStep('ANALYZING_PERFORMANCE');
            }
        };
        const timeoutId = setTimeout(() => speak(currentLine.line, handleAiTurnEnd), 700);
        return () => clearTimeout(timeoutId);
    }
  }, [step, script, currentTurn, selectedCharacter, speak, speechRecognizer, recognitionError, isAiSpeaking]);

  useEffect(() => () => {
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    if (speechRecognizer) speechRecognizer.abort();
    if (practiceRecognizer.current) practiceRecognizer.current.abort();
  }, [speechRecognizer]);

  const handleSceneSelect = (scene: Scene) => { setSelectedScene(scene); setStep('CHARACTER'); };
  const handleCharacterSelect = (character: string) => { setMistakes([]); setUserDialog([]); setSelectedCharacter(character); generateScript(selectedScene!, character); };
  const handleBackToScenes = () => { setStep('SCENE'); setSelectedCharacter(null); setSelectedScene(null); setCenteredScene('Shopkeeper and Customer'); };

  const pronounceWord = (text: string) => {
    if (!window.speechSynthesis || voices.length === 0) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voiceLang = language === 'hi' ? 'hi-IN' : 'en-IN';

    const preferredVoice = voices.find(v => v.lang === voiceLang && v.name.includes('Google')) || voices.find(v => v.lang === voiceLang);
    
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    } else {
      const fallbackVoice = voices.find(v => v.lang.startsWith(language)) || voices.find(v => v.lang.startsWith('en'));
      if (fallbackVoice) utterance.voice = fallbackVoice;
    }
    window.speechSynthesis.speak(utterance);
  };

  const backgroundStyle = useMemo(() => {
    if (step !== 'SCENE' && selectedScene) {
      return {
        backgroundImage: `url(${TALKERS_CAVE_SCENE_BACKGROUNDS[selectedScene]})`,
        backgroundSize: 'contain',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        backgroundColor: 'black',
      };
    }
    return {};
  }, [step, selectedScene]);

  const renderTitle = () => {
    switch (step) {
      case 'SCENE': return 'Select Scene';
      case 'CHARACTER': return 'Select Character';
      case 'LOADING_SCRIPT': return 'Creating Your Story...';
      case 'ANALYZING_PERFORMANCE': return 'Analyzing Your Performance...';
      case 'GAME': return '';
      case 'COMPLETE': return 'Great Job!';
      case 'PRACTICE_PREP': return 'Getting Practice Ready...';
      case 'PRACTICE': return "Let's Practice!";
      default: return '';
    }
  };

  const renderContent = () => {
    switch (step) {
      case 'SCENE': {
        const scenes = Object.keys(TALKERS_CAVE_SCENES) as Scene[];
        return <div className="flex-grow flex flex-col justify-center items-center w-full h-full overflow-hidden">
            <div className="flex w-full h-full items-center justify-center gap-4 md:gap-8 px-4">
              {scenes.map((scene) => (
                  <button key={scene} onClick={() => handleSceneSelect(scene)} onMouseEnter={() => setCenteredScene(scene)} aria-label={`Select scene: ${scene}`}
                    className={`relative w-60 md:w-72 aspect-[16/10] flex-shrink-0 overflow-hidden rounded-2xl shadow-lg transition-all duration-500 ease-in-out transform group ${centeredScene === scene ? 'scale-110 opacity-100 shadow-cyan-500/40 z-10' : 'scale-90 opacity-60'} hover:!scale-110 hover:!opacity-100 hover:shadow-cyan-400/50`}>
                    <img src={TALKERS_CAVE_SCENE_IMAGES[scene]} alt={scene} className="absolute inset-0 w-full h-full object-cover transition-transform duration-300 group-hover:scale-105" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent"></div>
                    <h3 className="absolute bottom-4 left-4 right-4 text-white font-bold text-lg md:text-xl text-left truncate" style={{ textShadow: '1px 1px 4px rgba(0,0,0,0.8)' }}>{scene}</h3>
                  </button>
              ))}
            </div>
        </div>;
      }
      case 'CHARACTER':
        if (!selectedScene) return null;
        return <div className="flex flex-col items-center justify-center h-full">
            {error && <p className="text-center text-red-400 mb-4 bg-black/50 p-2 rounded">{error}</p>}
            <div className="flex justify-center items-end gap-4 md:gap-8 flex-wrap">
              {TALKERS_CAVE_SCENES[selectedScene].map((character) => (
                  <button key={character} onClick={() => handleCharacterSelect(character)} className="flex flex-col items-center gap-4 transition-transform transform hover:scale-105 group">
                    <div className="w-36 h-72 md:w-48 md:h-96"><img src={TALKERS_CAVE_CHARACTER_IMAGES[character]} alt={character} className="w-full h-full object-contain" /></div>
                    <span className="text-lg md:text-xl font-bold px-4 py-2 rounded-lg bg-indigo-600 group-hover:bg-indigo-500 transition-colors">{character}</span>
                  </button>
                ))}
            </div>
        </div>;

      case 'LOADING_SCRIPT': return <div className="text-center text-slate-300 animate-pulse text-2xl">Please wait...</div>;
      case 'ANALYZING_PERFORMANCE': return <div className="text-center text-slate-300 animate-pulse text-2xl">Checking your work, one moment...</div>;
      case 'PRACTICE_PREP': return <div className="text-center text-slate-300 animate-pulse text-2xl">Analyzing words for practice...</div>;

      case 'GAME': {
        if (!script.length || !selectedCharacter || !selectedScene) return null;
        const [characterOnLeft, characterOnRight] = TALKERS_CAVE_SCENES[selectedScene];
        const currentLine = script[currentTurn];
        const isLeftCharacterSpeaking = currentLine.character === characterOnLeft;
        const isUserTurn = currentLine.character === selectedCharacter;

        return (
          <div className='w-full h-full relative flex flex-col overflow-hidden'>
            <div className="flex-grow relative flex items-end justify-center px-4 overflow-hidden">
              <div className={`absolute bottom-0 left-0 md:left-[5%] w-1/2 md:w-2/5 h-2/3 md:h-4/5 transition-transform duration-500 ${isLeftCharacterSpeaking ? 'scale-110' : 'scale-100'}`}><img src={TALKERS_CAVE_CHARACTER_IMAGES[characterOnLeft]} alt={characterOnLeft} className="w-full h-full object-contain"/></div>
              <div className={`absolute bottom-0 right-0 md:right-[5%] w-1/2 md:w-2/5 h-2/3 md:h-4/5 transition-transform duration-500 ${!isLeftCharacterSpeaking ? 'scale-110' : 'scale-100'}`}><img src={TALKERS_CAVE_CHARACTER_IMAGES[characterOnRight]} alt={characterOnRight} className="w-full h-full object-contain"/></div>
              <div className={`absolute top-[8%] w-4/5 md:w-2/5 max-w-lg transition-all duration-300 ease-out ${!isLeftCharacterSpeaking ? 'right-[5%] md:right-[15%]' : 'left-[5%] md:left-[15%]'}`}>
                <div className={`relative bg-white text-slate-900 p-4 rounded-2xl shadow-2xl ${!isLeftCharacterSpeaking ? 'rounded-br-none' : 'rounded-bl-none'}`}>
                  {isUserTurn ? (
                    <>
                      <p className="text-lg font-medium leading-relaxed">{currentLine.line}</p>
                      {(isRecognitionActive || userTranscript) && (
                        <>
                          <hr className="my-2 border-slate-200" />
                          <p className="text-lg font-medium leading-relaxed text-slate-500 min-h-[1.5em]">
                            {userTranscript || 'Listening...'}
                          </p>
                        </>
                      )}
                    </>
                  ) : <p className="text-lg font-medium">{currentLine.line}</p>}
                  <div className={`absolute bottom-0 h-0 w-0 border-solid border-transparent border-t-white ${!isLeftCharacterSpeaking ? 'right-4 border-r-[15px] border-l-0 border-t-[15px] -mb-[15px]' : 'left-4 border-l-[15px] border-r-0 border-t-[15px] -mb-[15px]'}`}></div>
                </div>
              </div>
            </div>
            <div className="h-16 flex-shrink-0 bg-slate-900/50 flex items-center justify-center text-slate-300 relative">
              {recognitionError ? <p className="text-red-400 font-semibold">{recognitionError}</p> : (isUserTurn ? (
                  <button
                      onClick={handleMicButtonClick}
                      disabled={isAiSpeaking}
                      className={`w-14 h-14 rounded-full flex items-center justify-center transition-all duration-300 transform 
                          ${isRecognitionActive ? 'bg-red-500 scale-110 animate-pulse' : 'bg-cyan-600 hover:bg-cyan-500 hover:scale-105'}
                          disabled:bg-slate-500 disabled:scale-100 disabled:cursor-not-allowed`}
                      aria-label={isRecognitionActive ? 'Stop recording' : 'Start recording'}
                  >
                      <MicrophoneIcon className="w-8 h-8 text-white" />
                  </button>
                ) : <p className="text-lg animate-pulse">{isAiSpeaking ? 'AI is speaking...' : 'AI is thinking...'}</p>)}
            </div>
          </div>
        );
      }
      case 'PRACTICE': {
          if (practiceWords.length === 0 || currentPracticeWordIndex >= practiceWords.length) {
              return <div className="text-center flex flex-col items-center justify-center h-full p-4">
                  <p className="text-xl text-slate-300 mb-8">No words to practice. Well done!</p>
                  <button onClick={() => onComplete(true)} className="px-8 py-4 bg-green-600 text-white font-bold rounded-lg text-xl hover:bg-green-700 transition-transform transform hover:scale-105">Finish</button>
              </div>;
          }
          const practiceItem = practiceWords[currentPracticeWordIndex];
          const getStatusMessage = () => {
              switch (practiceStatus) {
                  case 'LISTENING': return <p className="text-cyan-400 animate-pulse">Listening...</p>;
                  case 'SUCCESS': return <p className="text-green-400 font-bold">Great job!</p>;
                  case 'TRY_AGAIN': return <p className="text-red-400">Not quite. You said: <span className="font-bold">{practiceTranscript}</span>. Try again!</p>;
                  default: return <p className="text-slate-300">Click the mic and say the word.</p>;
              }
          };
          return <div className="w-full h-full flex flex-col items-center justify-center p-4 sm:p-8 animate-fade-in">
              <div className="w-full max-w-2xl text-center">
                  <div className="flex justify-center flex-wrap gap-4 mb-12">
                      {practiceItem.phonemes.map((phoneme, index) => (
                          <div key={index} className="bg-white rounded-2xl p-4 flex flex-col items-center justify-between gap-4 shadow-lg w-32">
                              <span className="text-purple-600 font-bold text-4xl sm:text-5xl" style={{minHeight: '48px'}}>{phoneme}</span>
                              <button onClick={() => pronounceWord(phoneme)} className="p-1" aria-label={`Listen to ${phoneme}`}>
                                  <PracticeSoundIcon />
                              </button>
                          </div>
                      ))}
                  </div>

                  <div className="flex flex-col items-center gap-4">
                      <button onClick={handlePracticeMicClick} disabled={practiceStatus === 'LISTENING' || practiceStatus === 'SUCCESS'}
                          className={`w-28 h-28 sm:w-32 sm:h-32 rounded-full flex items-center justify-center transition-all duration-300 transform 
                              ${practiceStatus === 'LISTENING' ? 'bg-cyan-500 scale-110 animate-pulse' : 'bg-purple-600 hover:bg-purple-500 hover:scale-105'}
                              disabled:bg-slate-500 disabled:scale-100 disabled:cursor-not-allowed`}>
                          <MicrophoneIcon className="w-12 h-12 sm:w-16 sm:h-16 text-white" />
                      </button>
                      <div className="h-8 text-xl mt-2">{getStatusMessage()}</div>
                  </div>
              </div>
              <button onClick={() => onComplete(true)} className="absolute bottom-6 right-6 text-slate-400 hover:text-white font-semibold transition-colors bg-black/30 px-4 py-2 rounded-lg">
                  Finish Practice
              </button>
          </div>
      }
      case 'COMPLETE': return <div className="text-center flex flex-col items-center justify-center h-full p-4">
            <p className="text-3xl sm:text-4xl text-green-400 mb-2">Perfect!</p>
            <p className="text-lg sm:text-xl text-slate-300 mb-8">You said everything correctly. Great job!</p>
            <button onClick={() => onComplete(true)} className="px-6 py-3 sm:px-8 sm:py-4 bg-green-600 text-white font-bold rounded-lg text-xl sm:text-2xl w-full max-w-xs hover:bg-green-700 transition-transform transform hover:scale-105">Finish</button>
        </div>;
      default: return null;
    }
  };

  return (
    <div
        className="w-full h-full text-white relative flex flex-col justify-center animate-fade-in"
        style={backgroundStyle}
    >
        <div className="absolute top-4 sm:top-6 left-1/2 -translate-x-1/2 w-full px-4 text-center z-20">
             <h1 className="text-3xl sm:text-5xl font-bold text-cyan-400" style={{textShadow: '2px 2px 8px rgba(0,0,0,0.7)'}}>{renderTitle()}</h1>
        </div>
        {step !== 'GAME' && <div className="absolute top-4 right-4 sm:top-6 sm:right-6 bg-slate-900/70 px-4 py-2 rounded-lg text-base sm:text-lg font-bold text-cyan-300 z-20 backdrop-blur-sm">
          Level: {currentLevel}
        </div>}
        {(step === 'SCENE' || step === 'CHARACTER') && (
            <button
                onClick={step === 'SCENE' ? onBackToGrades : handleBackToScenes}
                className="absolute top-4 left-4 sm:top-6 sm:left-6 text-slate-300 hover:text-white transition-colors z-20 font-bold flex items-center gap-2 text-sm sm:text-base"
            >
                <span className="text-xl sm:text-2xl">&larr;</span> {step === 'SCENE' ? 'Back to Grades' : 'Back'}
            </button>
        )}
        <div className="flex-grow flex flex-col justify-center overflow-hidden pt-20 sm:pt-24">
            {renderContent()}
        </div>
    </div>
  );
};
