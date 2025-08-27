#!/usr/bin/env python3
"""
Ceramatic 2.0 - Universal Setup & Launch Script
Cross-platform installation and launcher for all operating systems
"""

import os
import sys
import subprocess
import platform
import shutil
import venv
from pathlib import Path
import argparse
import urllib.request
import json
import time
from typing import Optional, Tuple, List

class CeramaticSetup:
    """Universal setup and launcher for Ceramatic 2.0"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.python_cmd = sys.executable
        self.venv_dir = Path(".venv")
        self.requirements_file = Path("requirements.txt")
        self.model_file = Path("Ceramatic_model_V1.pt")
        self.gradio_app = Path("ceramatic_gradio.py")
        self.model_url = "https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view"
        
        # Colors for terminal output
        self.colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'reset': '\033[0m'
        }
        
        # Disable colors on Windows CMD (not PowerShell/Terminal)
        if self.platform == 'windows' and not self.supports_color():
            self.colors = {k: '' for k in self.colors}
    
    def supports_color(self) -> bool:
        """Check if terminal supports colors"""
        if self.platform == 'windows':
            return os.environ.get('TERM') or 'WT_SESSION' in os.environ
        return True
    
    def print_header(self):
        """Print application header"""
        print(f"\n{self.colors['blue']}{'='*50}{self.colors['reset']}")
        print(f"{self.colors['blue']}     🏺 Ceramatic 2.0 - Setup Universale{self.colors['reset']}")
        print(f"{self.colors['blue']}{'='*50}{self.colors['reset']}\n")
    
    def print_status(self, message: str, status: str = "info"):
        """Print colored status message"""
        status_colors = {
            'info': self.colors['cyan'],
            'success': self.colors['green'],
            'warning': self.colors['yellow'],
            'error': self.colors['red']
        }
        status_labels = {
            'info': '[INFO]',
            'success': '[OK]',
            'warning': '[WARN]',
            'error': '[ERRORE]'
        }
        color = status_colors.get(status, self.colors['white'])
        label = status_labels.get(status, '[INFO]')
        print(f"{color}{label}{self.colors['reset']} {message}")
    
    def check_python_version(self) -> bool:
        """Check if Python version is 3.7+"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 7):
            self.print_status(f"Python {version.major}.{version.minor} trovato, richiesto 3.7+", "error")
            return False
        self.print_status(f"Python {version.major}.{version.minor}.{version.micro} trovato", "success")
        return True
    
    def check_device(self) -> Optional[str]:
        """Check device availability (CUDA, MPS, or CPU)"""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.print_status(f"GPU CUDA disponibile: {device_name} ({memory:.1f}GB)", "success")
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.print_status("Apple Silicon GPU (MPS) disponibile per accelerazione", "success")
                return "mps"
        except ImportError:
            pass
        except Exception as e:
            self.print_status(f"Errore controllo GPU: {e}", "warning")
        
        self.print_status("Nessuna GPU rilevata, verrà usata la CPU", "warning")
        return "cpu"
    
    def create_virtual_env(self) -> bool:
        """Create virtual environment"""
        if self.venv_dir.exists():
            self.print_status("Ambiente virtuale esistente trovato", "info")
            return True
        
        self.print_status("Creazione ambiente virtuale...", "info")
        try:
            venv.create(self.venv_dir, with_pip=True)
            self.print_status("Ambiente virtuale creato", "success")
            return True
        except Exception as e:
            self.print_status(f"Errore creazione ambiente virtuale: {e}", "error")
            return False
    
    def get_venv_python(self) -> str:
        """Get python executable path in virtual environment"""
        if self.platform == 'windows':
            python_exe = self.venv_dir / "Scripts" / "python.exe"
        else:
            python_exe = self.venv_dir / "bin" / "python"
        return str(python_exe)
    
    def get_venv_pip(self) -> str:
        """Get pip executable path in virtual environment"""
        if self.platform == 'windows':
            pip_exe = self.venv_dir / "Scripts" / "pip.exe"
        else:
            pip_exe = self.venv_dir / "bin" / "pip"
        return str(pip_exe)
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip in virtual environment"""
        self.print_status("Aggiornamento pip...", "info")
        pip_exe = self.get_venv_pip()
        try:
            subprocess.run(
                [pip_exe, "install", "--upgrade", "pip"],
                check=True,
                capture_output=True,
                text=True
            )
            self.print_status("pip aggiornato", "success")
            return True
        except subprocess.CalledProcessError as e:
            self.print_status(f"Errore aggiornamento pip: {e}", "error")
            return False
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        if not self.requirements_file.exists():
            self.print_status("File requirements.txt non trovato", "error")
            return False
        
        self.print_status("Installazione dipendenze (potrebbe richiedere alcuni minuti)...", "info")
        pip_exe = self.get_venv_pip()
        
        # Special handling for different platforms
        env = os.environ.copy()
        
        # macOS specific handling
        if self.platform == 'darwin':
            # Check for OpenBLAS
            if shutil.which('brew'):
                try:
                    result = subprocess.run(
                        ['brew', 'list', 'openblas'],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode != 0:
                        self.print_status("Installazione OpenBLAS per scipy...", "info")
                        subprocess.run(['brew', 'install', 'openblas'], check=False)
                        
                    # Set OpenBLAS paths
                    brew_prefix = subprocess.run(
                        ['brew', '--prefix', 'openblas'],
                        capture_output=True,
                        text=True
                    ).stdout.strip()
                    if brew_prefix:
                        env['OPENBLAS'] = brew_prefix
                        env['CFLAGS'] = f"-I{brew_prefix}/include"
                        env['LDFLAGS'] = f"-L{brew_prefix}/lib"
                except:
                    pass
            
            if platform.machine() == 'arm64':
                self.print_status("Apple Silicon rilevato, configurazione speciale...", "info")
                env['ARCHFLAGS'] = '-arch arm64'
        
        try:
            # First, ensure wheel and setuptools are installed
            subprocess.run(
                [pip_exe, "install", "--upgrade", "wheel", "setuptools"],
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
            
            # Install requirements
            process = subprocess.Popen(
                [pip_exe, "install", "-r", str(self.requirements_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            # Show progress
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output and ('Collecting' in output or 'Installing' in output or 'Successfully' in output):
                    print(f"  {output.strip()}")
            
            if process.returncode == 0:
                self.print_status("Dipendenze installate con successo", "success")
                return True
            else:
                self.print_status("Errore installazione dipendenze", "error")
                return False
                
        except Exception as e:
            self.print_status(f"Errore installazione dipendenze: {e}", "error")
            return False
    
    def check_model(self) -> bool:
        """Check if model file exists"""
        if self.model_file.exists():
            size_mb = self.model_file.stat().st_size / (1024 * 1024)
            self.print_status(f"Modello trovato: {self.model_file} ({size_mb:.1f}MB)", "success")
            return True
        
        self.print_status("Modello non trovato", "warning")
        print(f"\n{self.colors['yellow']}{'='*50}{self.colors['reset']}")
        print(f"{self.colors['yellow']}  ⚠️  ATTENZIONE: Modello richiesto!{self.colors['reset']}")
        print(f"{self.colors['yellow']}{'='*50}{self.colors['reset']}")
        print(f"\nScarica il modello da:")
        print(f"{self.colors['blue']}{self.model_url}{self.colors['reset']}")
        print(f"\nE salvalo come: {self.colors['cyan']}{self.model_file}{self.colors['reset']}")
        print(f"{self.colors['yellow']}{'='*50}{self.colors['reset']}\n")
        return False
    
    def launch_app(self, host: str = "127.0.0.1", port: int = 7860, share: bool = False):
        """Launch the Gradio application"""
        if not self.gradio_app.exists():
            self.print_status(f"File {self.gradio_app} non trovato", "error")
            return False
        
        print(f"\n{self.colors['green']}{'='*50}{self.colors['reset']}")
        print(f"{self.colors['green']}  🚀 Avvio Ceramatic 2.0 Interface{self.colors['reset']}")
        print(f"{self.colors['green']}{'='*50}{self.colors['reset']}\n")
        
        python_exe = self.get_venv_python()
        
        # Prepare launch command
        cmd = [python_exe, str(self.gradio_app)]
        
        # Add arguments if needed
        env = os.environ.copy()
        env['GRADIO_SERVER_NAME'] = host
        env['GRADIO_SERVER_PORT'] = str(port)
        if share:
            env['GRADIO_SHARE'] = 'true'
        
        print(f"Apertura browser su: {self.colors['cyan']}http://{host}:{port}{self.colors['reset']}")
        print(f"Per chiudere: premi {self.colors['yellow']}Ctrl+C{self.colors['reset']}\n")
        
        try:
            subprocess.run(cmd, env=env, check=True)
            return True
        except KeyboardInterrupt:
            print(f"\n{self.colors['yellow']}Applicazione interrotta dall'utente{self.colors['reset']}")
            return True
        except Exception as e:
            self.print_status(f"Errore avvio applicazione: {e}", "error")
            return False
    
    def full_setup(self) -> bool:
        """Run complete setup process"""
        self.print_header()
        
        # System info
        self.print_status(f"Sistema: {platform.system()} {platform.release()}", "info")
        self.print_status(f"Architettura: {platform.machine()}", "info")
        print()
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Create virtual environment
        if not self.create_virtual_env():
            return False
        
        # Upgrade pip
        if not self.upgrade_pip():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        print()
        
        # Check device (CUDA, MPS, or CPU)
        self.check_device()
        
        print()
        
        # Check model
        self.check_model()
        
        return True
    
    def run(self, skip_setup: bool = False, host: str = "127.0.0.1", 
            port: int = 7860, share: bool = False):
        """Main execution flow"""
        try:
            if not skip_setup:
                if not self.full_setup():
                    self.print_status("Setup non completato", "error")
                    return 1
            else:
                self.print_header()
                self.print_status("Skip setup, avvio diretto...", "info")
            
            # Launch application
            if not self.launch_app(host, port, share):
                return 1
            
            print(f"\n{self.colors['green']}Applicazione terminata con successo{self.colors['reset']}")
            return 0
            
        except KeyboardInterrupt:
            print(f"\n{self.colors['yellow']}Operazione annullata dall'utente{self.colors['reset']}")
            return 0
        except Exception as e:
            self.print_status(f"Errore imprevisto: {e}", "error")
            return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Ceramatic 2.0 - Setup e Launcher Universale"
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Salta il setup e avvia direttamente l'app"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host per il server Gradio (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Porta per il server Gradio (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Crea un link pubblico temporaneo (richiede internet)"
    )
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Esegui solo l'installazione senza avviare l'app"
    )
    
    args = parser.parse_args()
    
    setup = CeramaticSetup()
    
    if args.install_only:
        setup.full_setup()
        setup.print_status("Setup completato. Usa --skip-setup per avviare l'app", "success")
        return 0
    
    return setup.run(
        skip_setup=args.skip_setup,
        host=args.host,
        port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    sys.exit(main())