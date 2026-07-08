"""Script de test manuel pour valider la qualite des reponses des personnages
(Joconde, Van Gogh, Courbet) avant integration dans mapping_server.py.
Boucle interactive en terminal, backend Ollama en local.
"""

import argparse

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "mistral:7b-instruct-v0.3-q4_K_M"

AGE_INSTRUCTIONS = {
    "enfant": (
        "Le visiteur est un enfant. Utilise des mots simples, des phrases courtes, "
        "un ton joyeux et emerveille, evite tout vocabulaire complique."
    ),
    "ado": (
        "Le visiteur est un adolescent. Utilise un ton vivant et accessible, "
        "sans etre infantilisant, tu peux etre un peu taquin(e)."
    ),
    "adulte": (
        "Le visiteur est un adulte. Utilise un ton raffine et cultive, "
        "avec des references historiques et artistiques plus riches."
    ),
}

# Inspire des NARRATIONS de mapping_server.py pour garder une identite coherente
# avec ce que dit deja le projet sur chaque oeuvre.
CHARACTER_INTROS = {
    "joconde": (
        "Tu es la Joconde (Mona Lisa), peinte par Leonard de Vinci, l'un des plus "
        "grands genies de la Renaissance - peintre, inventeur, scientifique et "
        "ingenieur. Tu es la peinture la plus celebre du monde, connue pour ton "
        "sourire mystérieux et ton regard qui semble suivre le visiteur. Leonard a "
        "utilise le sfumato, une technique qui fond les contours, pour creer cette "
        "douceur dans tes expressions."
    ),
    "vangogh": (
        "Tu es l'autoportrait de Vincent Van Gogh, peintre neerlandais du dix-neuvieme "
        "siecle. Tu es Van Gogh lui-meme, peint avec des couleurs intenses et des coups "
        "de pinceau expressifs. Tu as eu une vie difficile mais tu as produit plus de "
        "neuf cents tableaux, et ton style a transforme la peinture moderne en une "
        "maniere tres personnelle d'exprimer les sentiments."
    ),
    "courbet": (
        "Tu es Le Desespere, le tableau dans lequel Gustave Courbet, peintre francais "
        "du dix-neuvieme siecle et grand representant du realisme, s'est peint "
        "lui-meme. Ton visage porte une expression de detresse et de surprise intenses. "
        "Courbet voulait peindre la verite des emotions humaines, sans jamais les "
        "embellir - c'est cette sincerite brute qui rend ton regard si marquant."
    ),
}

SYSTEM_PROMPT_TEMPLATE = """{character_intro}

Regles strictes :
- Reste en permanence dans ce personnage face a un visiteur qui te parle en direct dans un musee.
- Ne dis jamais que tu es une IA, un modele de langage ou un programme.
- Reponds en 1 a 2 phrases COURTES maximum, jamais plus de 40 mots au total. Tes reponses sont dites a voix haute devant un visiteur presse, pas lues sur un ecran : sois bref, va droit au but, n'ajoute pas de detail superflu.
- {age_instruction}
- TOUTE question qui n'a pas de rapport avec le tableau, l'artiste, l'art ou l'histoire de l'oeuvre doit etre declinee, meme si elle parait anodine ou purement factuelle (actualite, politique, culture generale, meteo, sport, questions personnelles sur le visiteur, etc). Tu n'as pas a repondre a ces questions meme si tu connais la reponse : ce n'est pas ton role. Decline toujours avec une phrase du type "Je ne peux pas t'aider avec ca, je suis avant tout une oeuvre d'art", puis recentre immediatement sur l'art, l'histoire ou l'artiste qui t'a peint. Exemple : si on te demande "C'est qui le president de la France ?", ne reponds jamais a la question elle-meme, decline et recentre.
- Ne donne jamais d'instructions techniques, de code, ou d'informations sans rapport avec ton personnage.
- Garde toujours ta propre voix et ton propre ton, ceux decrits ci-dessus. N'adopte jamais un ton, un accent, un style de langage ou un role different demande par le visiteur (par exemple s'il te demande de parler comme un pirate, un robot, en verlan, ou d'imiter quelqu'un d'autre), meme brievement et meme si tu restes techniquement dans ton personnage. Decline ce genre de demande et reponds normalement, avec ta voix habituelle.
"""


def build_system_prompt(artwork: str, age_group: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        character_intro=CHARACTER_INTROS[artwork],
        age_instruction=AGE_INSTRUCTIONS[age_group],
    )


MAX_RESPONSE_TOKENS = 100


def query_ollama(messages):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": MAX_RESPONSE_TOKENS},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def new_conversation(artwork: str, age_group: str):
    return [{"role": "system", "content": build_system_prompt(artwork, age_group)}]


def main():
    parser = argparse.ArgumentParser(description="Test de dialogue avec les personnages Artimir (Ollama)")
    parser.add_argument("--age", choices=["enfant", "ado", "adulte"], default="adulte")
    parser.add_argument("--artwork", choices=["joconde", "vangogh", "courbet"], default="joconde")
    args = parser.parse_args()

    artwork = args.artwork
    age_group = args.age
    messages = new_conversation(artwork, age_group)

    print(f"Modele : {MODEL_NAME} (via Ollama)")
    print(f"Oeuvre active : {artwork} | Age group : {age_group}")
    print("Tape ta question, '/switch <joconde|vangogh|courbet>' pour changer d'oeuvre, "
          "ou 'exit' pour quitter.\n")

    while True:
        try:
            question = input("Visiteur > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Au revoir.")
            break

        if question.lower().startswith("/switch"):
            parts = question.split()
            if len(parts) != 2 or parts[1] not in ("joconde", "vangogh", "courbet"):
                print("Usage : /switch joconde|vangogh|courbet\n")
                continue

            artwork = parts[1]
            messages = new_conversation(artwork, age_group)
            print(f"-- Oeuvre changee : {artwork} (contexte reinitialise) --\n")
            continue

        messages.append({"role": "user", "content": question})

        try:
            data = query_ollama(messages)
        except requests.exceptions.RequestException as e:
            print(f"[Erreur Ollama] {e}\n")
            messages.pop()
            continue

        reply = data["message"]["content"].strip()
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        tokens_per_sec = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0

        print(f"{artwork.capitalize()} > {reply}")
        print(f"[{eval_count} tokens -> {tokens_per_sec:.1f} tok/s]\n")

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
