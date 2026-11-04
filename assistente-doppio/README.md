# Assistente Doppio

O Assistente Doppio é o assistente de IA dos alunos da mentoria Doppio, do Sérgio Nogueira. Faz Reels, carrosséis e legendas com a metodologia Doppio, a partir das tuas fotos, e responde com o que o Sérgio ensina nas aulas.

Só funciona para alunos com a mentoria ativa no Lato.

## Instalar (uma vez)

**1. O uv**, que instala sozinho o resto do que o motor precisa.

No Mac, abre o Terminal e cola:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

No Windows, abre o PowerShell e cola:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. O plugin**, dentro do Claude Code:
```
/plugin marketplace add lato-studio/assistente-doppio
/plugin install assistente-doppio@doppio
```

**3. As atualizações automáticas.** Em `/plugin`, abre o marketplace `doppio` e liga as atualizações automáticas. Assim recebes as melhorias do motor sem fazer nada. As regras do método atualizam-se sempre sozinhas, porque vêm do Assistente.

**4. Entrar.** Na primeira vez, o Assistente pede o email com que entras no Lato e manda-te um código. O login dura uma semana.

## Usar

Abre a pasta do teu negócio no Claude Code e diz, por exemplo:
- "Vamos começar" (o primeiro dia: a tua ficha de negócio);
- "Quero fazer um Reel de prova";
- "Faz o banco de slides deste casamento";
- "Escreve a legenda deste carrossel".

Na primeira vez que o motor corre, demora uns minutos a instalar as bibliotecas.

## O que é teu e o que não é

As fotos, os vídeos, a tua ficha e os conteúdos que crias são teus. As regras e o método que o Assistente usa são da mentoria Doppio, licenciados a ti enquanto fores aluno.
