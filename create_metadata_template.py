#!/usr/bin/env python3
"""
Script per creare un template Excel per i metadata di Ceramatic 2.0
"""

import pandas as pd
import os
from pathlib import Path
import sys

def create_metadata_template(imgs_dir=None, output_file="metadata_template.xlsx"):
    """
    Crea un template Excel per i metadata.
    
    Args:
        imgs_dir: Directory con le immagini (opzionale, per pre-compilare TAV)
        output_file: Nome del file output
    """
    
    print("🏺 Ceramatic 2.0 - Creazione Template Metadata")
    print("=" * 50)
    
    data = []
    
    if imgs_dir and os.path.exists(imgs_dir):
        # Se fornita una directory, pre-compila con i nomi delle immagini
        print(f"📂 Scansione immagini in: {imgs_dir}")
        
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        img_files = []
        
        for file in sorted(os.listdir(imgs_dir)):
            if Path(file).suffix.lower() in img_extensions:
                img_files.append(Path(file).stem)
        
        if img_files:
            print(f"✅ Trovate {len(img_files)} immagini")
            
            # Chiedi quanti frammenti per immagine
            print("\n📝 Configurazione frammenti per immagine:")
            print("(Premi ENTER per 1 frammento, o inserisci il numero)")
            
            for img_name in img_files:
                try:
                    num_input = input(f"  {img_name}.jpg - Numero frammenti [1]: ").strip()
                    num_fragments = int(num_input) if num_input else 1
                except ValueError:
                    num_fragments = 1
                
                # Crea righe per ogni frammento
                for i in range(num_fragments):
                    inv_number = 1000 + len(data) + 1  # Numero inventario progressivo
                    data.append({
                        'TAV': img_name,
                        'INV': inv_number,
                        'DIAM': 0.0,  # Da compilare
                        'FLIP': 0
                    })
            
            print(f"\n✅ Create {len(data)} righe nel template")
        else:
            print("⚠️ Nessuna immagine trovata nella directory")
            
    else:
        # Template vuoto con esempi
        print("📝 Creazione template vuoto con esempi...")
        
        data = [
            {'TAV': 'esempio_001', 'INV': 1001, 'DIAM': 18.5, 'FLIP': 0},
            {'TAV': 'esempio_001', 'INV': 1002, 'DIAM': 22.0, 'FLIP': 0},
            {'TAV': 'esempio_002', 'INV': 1003, 'DIAM': 15.0, 'FLIP': 1},
            {'TAV': 'esempio_003', 'INV': 1004, 'DIAM': 0.0, 'FLIP': 0},
            {'TAV': '', 'INV': '', 'DIAM': '', 'FLIP': ''},  # Riga vuota per aggiungere
        ]
    
    # Crea DataFrame
    df = pd.DataFrame(data)
    
    # Salva in Excel con formattazione
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Metadata', index=False)
        
        # Ottieni il worksheet per formattazione
        worksheet = writer.sheets['Metadata']
        
        # Imposta larghezza colonne
        worksheet.column_dimensions['A'].width = 15  # TAV
        worksheet.column_dimensions['B'].width = 10  # INV  
        worksheet.column_dimensions['C'].width = 10  # DIAM
        worksheet.column_dimensions['D'].width = 8   # FLIP
        
        # Aggiungi commenti alle intestazioni
        from openpyxl.comments import Comment
        
        worksheet['A1'].comment = Comment(
            'Nome del file immagine SENZA estensione\nEs: se il file è 149.jpg, inserire solo 149',
            'Ceramatic'
        )
        worksheet['B1'].comment = Comment(
            'Numero inventario del frammento\nDeve essere un numero intero univoco',
            'Ceramatic'
        )
        worksheet['C1'].comment = Comment(
            'Diametro in cm per ricostruzione\nUsare 0 per non ricostruire il profilo completo',
            'Ceramatic'
        )
        worksheet['D1'].comment = Comment(
            'Inversione orizzontale\n0 = Non invertire\n1 = Inverti',
            'Ceramatic'
        )
    
    print(f"\n✅ Template salvato: {output_file}")
    print("\n📋 Prossimi passi:")
    print("1. Apri il file Excel")
    print("2. Compila i valori DIAM con i diametri misurati")
    print("3. Modifica FLIP se necessario (1 per invertire)")
    print("4. Salva e usa il file in Ceramatic")
    
    return output_file

def main():
    """Entry point principale"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Crea un template Excel per i metadata di Ceramatic 2.0"
    )
    parser.add_argument(
        "--imgs-dir",
        help="Directory contenente le immagini (per pre-compilare TAV)"
    )
    parser.add_argument(
        "--output",
        default="metadata_template.xlsx",
        help="Nome file output (default: metadata_template.xlsx)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modalità veloce: assume 1 frammento per immagine"
    )
    
    args = parser.parse_args()
    
    if args.quick and args.imgs_dir:
        # Modalità veloce: crea automaticamente con 1 frammento per immagine
        print("⚡ Modalità veloce: 1 frammento per immagine")
        
        if os.path.exists(args.imgs_dir):
            img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            data = []
            
            for file in sorted(os.listdir(args.imgs_dir)):
                if Path(file).suffix.lower() in img_extensions:
                    img_name = Path(file).stem
                    inv_number = 1000 + len(data) + 1
                    data.append({
                        'TAV': img_name,
                        'INV': inv_number,
                        'DIAM': 0.0,
                        'FLIP': 0
                    })
            
            if data:
                df = pd.DataFrame(data)
                df.to_excel(args.output, index=False)
                print(f"✅ Template creato con {len(data)} righe: {args.output}")
            else:
                print("⚠️ Nessuna immagine trovata")
        else:
            print(f"❌ Directory non trovata: {args.imgs_dir}")
    else:
        # Modalità interattiva
        create_metadata_template(args.imgs_dir, args.output)

if __name__ == "__main__":
    main()