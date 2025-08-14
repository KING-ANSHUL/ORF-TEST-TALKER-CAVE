import React from 'react';
import { MicrophoneIcon } from './icons/MicrophoneIcon';

interface PermissionScreenProps {
  onPermissionGranted: () => void;
  onBack: () => void;
}

export const PermissionScreen: React.FC<PermissionScreenProps> = ({ onPermissionGranted, onBack }) => {
  const [status, setStatus] = React.useState<'idle' | 'requesting' | 'denied'>('idle');

  const handleRequestPermission = async () => {
    setStatus('requesting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());
      console.log("Microphone access granted!");
      onPermissionGranted();
    } catch (err) {
      console.error("Microphone access denied:", err);
      setStatus('denied');
    }
  };

  return (
    <div className="w-full h-full text-white flex flex-col items-center justify-center p-8 animate-fade-in text-center">
      <button
        onClick={onBack}
        className="absolute top-4 left-4 sm:top-6 sm:left-6 text-slate-300 hover:text-white transition-colors z-20 font-bold flex items-center gap-2 text-sm sm:text-base"
      >
        <span className="text-xl sm:text-2xl">&larr;</span> Back to Language
      </button>

      <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-8 max-w-lg w-full shadow-lg">
        <MicrophoneIcon className="w-16 h-16 text-cyan-400 mx-auto mb-6" />
        <h2 className="text-3xl font-bold mb-4">Microphone Access</h2>
        <p className="text-slate-300 text-lg mb-8">
          This app needs access to your microphone so it can hear you read and help you practice your speaking.
        </p>

        {status !== 'denied' ? (
          <button
            onClick={handleRequestPermission}
            disabled={status === 'requesting'}
            className="w-full px-8 py-4 bg-cyan-500 text-white font-bold rounded-lg text-xl hover:bg-cyan-600 transition-transform transform hover:scale-105 disabled:bg-slate-600 disabled:cursor-wait"
          >
            {status === 'requesting' ? 'Waiting for permission...' : 'Allow Access'}
          </button>
        ) : (
          <div className="bg-red-900/50 border border-red-500 text-red-300 rounded-lg p-4 text-left">
            <h3 className="font-bold mb-2">Permission Denied</h3>
            <p>
              You have denied microphone access. To use this app, you'll need to grant permission from your browser's address bar or settings.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
