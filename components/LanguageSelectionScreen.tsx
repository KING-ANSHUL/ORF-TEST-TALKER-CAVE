import React from 'react';

interface LanguageSelectionScreenProps {
  onLanguageSelect: (language: string) => void;
  onBack: () => void;
}

export const LanguageSelectionScreen: React.FC<LanguageSelectionScreenProps> = ({ onLanguageSelect, onBack }) => {
  const languages = [
    { code: 'en', name: 'English' },
    { code: 'hi', name: 'Hindi' },
  ];

  return (
    <div className="w-full h-full text-white flex flex-col items-center justify-center p-8 animate-fade-in text-center">
      <button
        onClick={onBack}
        className="absolute top-4 left-4 sm:top-6 sm:left-6 text-slate-300 hover:text-white transition-colors z-20 font-bold flex items-center gap-2 text-sm sm:text-base"
      >
        <span className="text-xl sm:text-2xl">&larr;</span> Back to Grades
      </button>
      <h2 className="text-2xl sm:text-3xl font-semibold text-slate-200 mb-12">
        Please select a language.
      </h2>
      <div className="flex gap-4 sm:gap-6">
        {languages.map((lang) => (
          <button
            key={lang.code}
            onClick={() => onLanguageSelect(lang.code)}
            className="w-32 h-32 sm:w-36 sm:h-36 bg-slate-800/70 rounded-2xl flex flex-col items-center justify-center text-slate-200 hover:bg-cyan-600 hover:text-white hover:scale-110 transition-all duration-300 shadow-lg backdrop-blur-sm"
          >
            <span className="text-3xl sm:text-4xl font-bold">{lang.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
