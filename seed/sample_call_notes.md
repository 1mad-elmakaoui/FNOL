# Sample call — script to record your own test audio

No audio ships with the repo (real calls contain personal data). To try the full
pipeline (upload → local transcription → extraction → click-to-seek), record
yourself and a colleague reading the script below — phone, laptop mic, anything.
Two different voices (or two stereo channels) help the diarization fallback.

The values match `seed/policies.csv`, so `lookup_policy` will find the policy.
Line 8 contains a deliberate self-correction: the extracted plate must be
`1234-B-56`, not `1234-A-56`.

| # | Speaker | Line |
|---|---------|------|
| 1 | Agent  | Horizon Assurance, good morning, this is Karim. How can I help? |
| 2 | Caller | Hello, I had an accident yesterday in Casablanca and I need to declare it. |
| 3 | Agent  | Of course. Your policy number please? |
| 4 | Caller | F N X, three seven seven one zero two. |
| 5 | Agent  | Thank you. And your full name? |
| 6 | Caller | Sarah Benali. |
| 7 | Agent  | What is the registration of your car? |
| 8 | Caller | The plate is 1234-A-56… no sorry, 1234-B-56. B like Beirut. |
| 9 | Agent  | When did it happen? |
| 10 | Caller | Yesterday afternoon, around three… actually no, closer to half past four. |
| 11 | Agent  | Where was it? |
| 12 | Caller | On Boulevard Zerktouni, near the Twin Center. |
| 13 | Agent  | What damage is there on your car? |
| 14 | Caller | The front left wing is pushed in and the left headlight is smashed. |
| 15 | Agent  | Was anyone injured? |
| 16 | Caller | No, nobody was hurt. |
| 17 | Agent  | Was another vehicle involved? |
| 18 | Caller | Yes, a taxi. The driver is Hassan Ouazzani, plate 56789-A-6, insured with Wafa Assurance. |
| 19 | Agent  | Did the police come? |
| 20 | Caller | No, we filled in a constat together. |
| 21 | Agent  | Any witnesses? |
| 22 | Caller | No witnesses. |
| 23 | Agent  | And a phone number to reach you? |
| 24 | Caller | 06 61 23 45 76. |

Expected result: every required field filled and nothing missing (no police
report was filed, so no report number is needed; witnesses = none).
`vehicle_plate` = `1234-B-56` with its evidence pointing at line 8, and clicking
it plays the correction.

## French variant (tests language detection + French prompts)

> **Conseiller :** Horizon Assurance, service sinistres, bonjour.
> **Appelante :** Bonjour, je déclare un accrochage d'hier. Mon contrat c'est FNX-640075, Claire Dubois.
> **Conseiller :** Quelle est l'immatriculation ?
> **Appelante :** GH-512-KL. C'était hier vers dix-huit heures quinze, sur le parking de la Part-Dieu à Lyon.
> **Conseiller :** Des blessés ?
> **Appelante :** Non, personne. La portière arrière droite est enfoncée. L'autre conducteur, Bernard Petit, plaque CX-208-VN, assuré chez Groupama.
> **Conseiller :** Un numéro pour vous rappeler ?
> **Appelante :** Le 06 23 45 67 89.

Missing on purpose: police report, witnesses → they appear in the follow-up
call script, in French.

## Tips

* `WHISPER_MODEL=small` is the CPU default; `medium`/`large-v3` are noticeably
  better on noisy phone audio if you have the time or a GPU.
* Set `HF_TOKEN` (after accepting the pyannote model terms on Hugging Face) for
  proper diarization; otherwise the worker logs that it uses the energy-based
  fallback.
* Uploads are processed by the `worker` container: `make logs` to follow.
