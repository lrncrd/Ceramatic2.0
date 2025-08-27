# 📊 Guida ai Metadata per Ceramatic 2.0

## Formato del File

Il file metadata può essere in formato **Excel (.xlsx)** o **CSV (.csv)** e deve contenere 4 colonne obbligatorie:

## Struttura delle Colonne

### 1. **TAV** (Tavola/Immagine)
- **Tipo**: Testo o Numero
- **Descrizione**: Nome del file immagine SENZA estensione
- **Esempio**: Se il file si chiama `149.jpg`, inserisci solo `149`
- **Note**: Deve corrispondere esattamente al nome del file

### 2. **INV** (Inventario)
- **Tipo**: Numero intero
- **Descrizione**: Numero di inventario del frammento ceramico
- **Esempio**: `1001`, `1002`, `1003`
- **Note**: Verrà visualizzato sull'immagine finale
- **Ordine**: Se ci sono più frammenti nella stessa immagine, l'ordine delle righe determina l'assegnazione (da sinistra a destra nell'immagine)

### 3. **DIAM** (Diametro)
- **Tipo**: Numero decimale (in centimetri)
- **Descrizione**: Diametro ricostruito del vaso
- **Esempio**: `15.5`, `22.0`, `0`
- **Note**: 
  - Usa `0` o lascia vuoto se non vuoi ricostruire il profilo completo
  - Se > 0, il software ricostruirà il profilo simmetrico del vaso

### 4. **FLIP** (Inversione)
- **Tipo**: Numero (0 o 1)
- **Descrizione**: Inverte orizzontalmente il profilo del frammento
- **Valori**:
  - `0` = Non invertire (default)
  - `1` = Inverti orizzontalmente
- **Quando usare**: Se il frammento è stato scansionato al contrario

## Esempio di File Excel

| TAV | INV | DIAM | FLIP |
|-----|-----|------|------|
| 146 | 1001 | 18.5 | 0 |
| 146 | 1002 | 22.0 | 0 |
| 147 | 1003 | 15.0 | 1 |
| 148 | 1004 | 0 | 0 |
| 149 | 1005 | 25.5 | 0 |
| 149 | 1006 | 18.0 | 0 |
| 149 | 1007 | 0 | 0 |

## Casi Speciali

### 📸 Immagini con Multipli Frammenti

Se un'immagine contiene più frammenti ceramici:
1. Crea una riga per ogni frammento
2. Usa lo stesso valore TAV per tutte le righe
3. L'ordine delle righe determina l'assegnazione (da sinistra a destra)

**Esempio**: L'immagine `149.jpg` contiene 3 frammenti:
| TAV | INV | DIAM | FLIP |
|-----|-----|------|------|
| 149 | 2001 | 18.5 | 0 | ← Frammento più a sinistra
| 149 | 2002 | 22.0 | 0 | ← Frammento centrale
| 149 | 2003 | 15.0 | 0 | ← Frammento più a destra

### 🔄 Solo Estrazione Senza Ricostruzione

Per estrarre solo il frammento senza ricostruire il profilo completo:
- Imposta `DIAM = 0` o lascia vuoto

### ⚠️ Immagini Senza Corrispondenza

Se il numero di frammenti rilevati non corrisponde al numero di righe nel metadata:
- L'immagine verrà saltata con un avviso
- Controlla che il numero di righe corrisponda ai frammenti visibili

## Come Creare il File

### Metodo 1: Excel
1. Apri Microsoft Excel o LibreOffice Calc
2. Crea le 4 colonne: TAV, INV, DIAM, FLIP
3. Inserisci i dati
4. Salva come `.xlsx`

### Metodo 2: CSV
1. Apri un editor di testo
2. Prima riga: `TAV,INV,DIAM,FLIP`
3. Righe successive: valori separati da virgola
4. Salva con estensione `.csv`

**Esempio CSV**:
```csv
TAV,INV,DIAM,FLIP
146,1001,18.5,0
146,1002,22.0,0
147,1003,15.0,1
```

### Metodo 3: Google Sheets
1. Crea il file in Google Sheets
2. File → Scarica → Microsoft Excel (.xlsx)

## Template Pronto

Puoi usare il file di esempio:
- `demo/metadata_example.xlsx`

Copia questo file e modifica con i tuoi dati.

## Validazione

Prima di elaborare, verifica:
- ✅ I nomi in TAV corrispondono ai file immagine (senza estensione)
- ✅ Tutti gli INV sono numeri univoci
- ✅ I DIAM sono numeri (usa punto per decimali: `15.5` non `15,5`)
- ✅ FLIP contiene solo 0 o 1
- ✅ Il numero di righe per ogni TAV corrisponde ai frammenti nell'immagine

## Errori Comuni e Soluzioni

| Errore | Causa | Soluzione |
|--------|-------|-----------|
| "No tabular data found" | Nome TAV non corrisponde | Verifica che il nome sia identico (senza estensione) |
| "diameter is not a number" | DIAM contiene testo | Usa solo numeri o lascia vuoto |
| "Number of masks does not match" | Righe ≠ frammenti rilevati | Aggiungi/rimuovi righe per corrispondere |
| "Missing required columns" | Colonne mancanti o rinominate | Usa esattamente: TAV, INV, DIAM, FLIP |

## Suggerimenti

1. **Organizzazione**: Numera gli INV in modo progressivo per facilità di riferimento
2. **Backup**: Mantieni una copia del file metadata originale
3. **Test**: Prova prima con poche immagini per verificare il risultato
4. **Diametri**: Misura i diametri archeologicamente prima di inserirli
5. **Controllo Qualità**: Verifica visivamente il primo batch prima di elaborare tutto

## Esempio Pratico

Per elaborare questa struttura di file:
```
pottery_photos/
├── shard_001.jpg (2 frammenti)
├── shard_002.jpg (1 frammento)
└── shard_003.jpg (3 frammenti)
```

Il file metadata dovrebbe essere:
| TAV | INV | DIAM | FLIP |
|-----|-----|------|------|
| shard_001 | 101 | 12.5 | 0 |
| shard_001 | 102 | 15.0 | 0 |
| shard_002 | 103 | 18.5 | 1 |
| shard_003 | 104 | 0 | 0 |
| shard_003 | 105 | 22.0 | 0 |
| shard_003 | 106 | 25.5 | 0 |