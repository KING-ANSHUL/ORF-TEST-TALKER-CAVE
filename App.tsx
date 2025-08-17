import React from 'react';
import { TalkersCaveGame } from './components/TalkersCaveGame';
import { GradeSelectionScreen } from './components/GradeSelectionScreen';
import { LanguageSelectionScreen } from './components/LanguageSelectionScreen';
import { PermissionScreen } from './components/PermissionScreen';
import { TALKERS_CAVE_SCENE_IMAGES, TALKERS_CAVE_CHARACTER_IMAGES, TALKERS_CAVE_SCENE_BACKGROUNDS } from './constants';

const preloadImages = (urls: string[]) => {
  urls.forEach(url => {
    const img = new Image();
    img.src = url;
  });
};

export const App: React.FC = () => {
  const [userGrade, setUserGrade] = React.useState<number | null>(null);
  const [selectedLanguage, setSelectedLanguage] = React.useState<string | null>(null);
  const [micPermissionGranted, setMicPermissionGranted] = React.useState(false);
  const [currentLevel, setCurrentLevel] = React.useState(1);
  const [gameKey, setGameKey] = React.useState(0);
  const [isFinished, setIsFinished] = React.useState(false);

  React.useEffect(() => {
    const initialImages = [
      '/Background.png',
      ...Object.values(TALKERS_CAVE_SCENE_IMAGES),
      ...Object.values(TALKERS_CAVE_CHARACTER_IMAGES),
      ...Object.values(TALKERS_CAVE_SCENE_BACKGROUNDS),
    ];
    preloadImages(initialImages);
  }, []);

  React.useEffect(() => {
    if (selectedLanguage && !micPermissionGranted) {
      navigator.permissions.query({ name: 'microphone' as PermissionName }).then((permissionStatus) => {
        if (permissionStatus.state === 'granted') {
          setMicPermissionGranted(true);
        }
      });
    }
  }, [selectedLanguage, micPermissionGranted]);

  const handleSetGrade = (grade: number) => {
    setUserGrade(grade);
    setCurrentLevel(1);
    setGameKey(0);
    setIsFinished(false);
    setSelectedLanguage(null);
    setMicPermissionGranted(false);
  };

  const handleLanguageSelect = (language: string) => {
    setSelectedLanguage(language);
  };

  const handleBackToGrades = () => {
    setUserGrade(null);
    setSelectedLanguage(null);
    setMicPermissionGranted(false);
  };

  const handleFinish = (success: boolean) => {
    setCurrentLevel(prevLevel => prevLevel + 1);
    setIsFinished(true);
  };

  const handlePlayAgain = () => {
    setIsFinished(false);
    setGameKey(prevKey => prevKey + 1);
  };

  const handlePermissionGranted = () => {
    setMicPermissionGranted(true);
  };

  return (
    <main className="h-screen w-screen text-white font-sans flex flex-col select-none relative">
      <div className="flex-1 flex flex-col overflow-hidden">
        {!userGrade ? (
          <GradeSelectionScreen onGradeSelect={handleSetGrade} />
        ) : !selectedLanguage ? (
          <LanguageSelectionScreen onLanguageSelect={handleLanguageSelect} onBack={() => setUserGrade(null)} />
        ) : !micPermissionGranted ? (
          <PermissionScreen onPermissionGranted={handlePermissionGranted} onBack={() => setSelectedLanguage(null)} />
        ) : (
          <div className="relative flex-1 overflow-hidden">
            {isFinished ? (
              <div className="absolute inset-0 backdrop-blur-sm flex flex-col justify-center items-center z-10 animate-fade-in text-center p-4">
                <h2 className="text-4xl sm:text-5xl font-bold text-cyan-400 mb-2" style={{textShadow: '2px 2px 8px rgba(0,0,0,0.7)'}}>
                  Level {currentLevel - 1} Complete!
                </h2>
                <p className="text-xl sm:text-2xl text-white mb-8" style={{textShadow: '2px 2px 8px rgba(0,0,0,0.7)'}}>Next Level: {currentLevel}</p>
                <button
                  onClick={handlePlayAgain}
                  className="px-8 py-4 bg-cyan-500 text-white font-bold rounded-lg text-xl sm:text-2xl hover:bg-cyan-600 transition-transform transform hover:scale-105"
                >
                  Start Level {currentLevel}
                </button>
              </div>
            ) : (
              <TalkersCaveGame
                key={gameKey}
                onComplete={handleFinish}
                userGrade={userGrade}
                currentLevel={currentLevel}
                onBackToGrades={handleBackToGrades}
                language={selectedLanguage!}
              />
            )}
          </div>
        )}
      </div>
    </main>
  );
};
