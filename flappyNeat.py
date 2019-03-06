from itertools import cycle
import configparser
import random
import sys

import pygame
from pygame.locals import *


FPS = 200
SCREENWIDTH  = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}
PARSER = configparser.ConfigParser()

PARSER.read('configs.txt')
game_settings = PARSER['GAME']
if 'FPS' in game_settings:
    FPS = int(game_settings['FPS'])
if 'PIPEGAPSIZE' in game_settings:
    PIPEGAPSIZE = int(game_settings['PIPEGAPSIZE'])

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


try:
    xrange
except NameError:
    xrange = range


def main(network=None):
    """
    Main function can accept a Neat Network object and determine how
    well the network can play flappy bird.

    Parameters
    ----------
    network (neat network object): a network with 8 inputs and one output

    Returns
    -------
    totTime : Total time the network survived

    """
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        # Play game
        crashInfo = mainGame(network=network)

        # Get total time
        totTime = pygame.time.get_ticks()
        #print(crashInfo['score'],  end='\t')

        return crashInfo['score']


def mainGame(network=None):
    pipegapsize = PIPEGAPSIZE
    show_score = True

    movementInfo = {
                    'playery': int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2),
                    'basex': 0,
                    'playerIndexGen': cycle([0, 1, 2, 1]),
                }

    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe(pipegapsize)
    newPipe2 = getRandomPipe(pipegapsize)

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerRot     =  45   # player's rotation
    playerVelRot  =   3   # angular speed
    playerRotThr  =  20   # rotation threshold
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = False # True when player flaps


    while True:
        for event in pygame.event.get():

            # check for player exiting
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

            # check for player flapping
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery > -2 * IMAGES['player'][0].get_height():
                    playerVelY = playerFlapAcc
                    playerFlapped = True

                    #SOUNDS['wing'].play()



        # check for crash here
        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)
        if crashTest[0]:
            return {
                'y': playery,
                'x': playerx,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': playerVelY,
                'playerRot': playerRot,
            }

        # find the four closest pipes
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2

        for i in range(len(upperPipes)):
            if playerMidPos < upperPipes[i]['x'] + IMAGES['pipe'][0].get_width():
                # next two pipe x pos
                uPipe1X = upperPipes[i]['x']
                uPipe2X = upperPipes[i]['x']

                # next two pipe y pos
                uPipe1Y = upperPipes[i]['y']
                uPipe2Y = upperPipes[i+1]['y']

                break

        for i in range(len(lowerPipes)):
            if playerMidPos < lowerPipes[i]['x'] + IMAGES['pipe'][0].get_width():
                # next two pipe x pos
                lPipe1X = lowerPipes[i]['x']
                lPipe2X = lowerPipes[i]['x']

                # next two pipe y pos
                lPipe1Y = lowerPipes[i]['y']
                lPipe2Y = lowerPipes[i+1]['y']

                break

        # NETWORK DECIDES TO FLAP OR NOT
        # print(playery)
        # print("Net output: "+str(network.activate([playery,0])))
        input_vals = [playery, playerVelY,
                      uPipe1X, uPipe1Y,
                      lPipe1X, lPipe1Y,
                      uPipe2X, uPipe2Y,
                      lPipe2X, lPipe2Y]

        input_dict = {'Y': playery,
                      'dY': playerVelY,
                      'E1':'',
                      'E2':'',
                      'P1UX': uPipe1X,
                      'P1LX': lPipe1X,
                      'P1UY': uPipe1Y,
                      'P1LY': lPipe1Y,
                      'P2UX': uPipe2X,
                      'P2LX': lPipe2X,
                      'P2UY': uPipe2Y,
                      'P2LY': lPipe2Y,
                      }
        input_dict = {'{:>4}'.format(k):'{:>5}'.format(v) for k,v in input_dict.items()}

        """
        NEURAL NETWORK MAKES A MOVE HERE

        Inputs
        ------
        playery    : bird height
        playerVelY : player velocity

        uPipe1Y    : the vertical distance from closest upper pipe
        uPipe1X    : the horizontal distance from closest upper pipe

        uPipe2Y    : the vertical distance from second closest upper pipe
        uPipe2X    : the horizontal distance from second closest upper pipe

        lPipe1Y    : the vertical distance from closest lower pipe
        lPipe1X    : the horizontal distance from closest lower pipe

        lPipe2Y    : the vertical distance from second closest lower pipe
        lPipe2X    : the horizontal distance from second closest lower pipe

        Outputs
        -------
        If output is greater than .5 --> FLAP
        if output is .5 or less      --> DO NOTHING
        """

        if network is not None:
            if network.activate(input_vals) > .5:

                if playery > -2 * IMAGES['player'][0].get_height():
                    playerVelY = playerFlapAcc
                    playerFlapped = True
                    # SOUNDS['wing'].play()

        """ END NEURAL NET CODE"""




        # check for score
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 50
                SOUNDS['point'].play()
        score += 1

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # rotate the player
        if playerRot > -90:
            playerRot -= playerVelRot

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            playerRot = 45

        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is close to left of screen
        if 25 < upperPipes[0]['x'] < 30:
            newPipe = getRandomPipe(pipegapsize)
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score

        if show_score:
            showScore(score)
            showData(input_dict)


        # Player rotation has a threshold
        visibleRot = playerRotThr
        if playerRot <= playerRotThr:
            visibleRot = playerRot

        playerSurface = pygame.transform.rotate(IMAGES['player'][playerIndex], visibleRot)
        SCREEN.blit(playerSurface, (playerx, playery))

        pygame.display.update()
        while True:
            try:
                PARSER.read('configs.txt')
                game_settings = PARSER['GAME']
                if 'FPS' in game_settings:
                    FPS = int(game_settings['FPS'])
                if 'PIPEGAPSIZE' in game_settings:
                    pipegapsize = int(game_settings['PIPEGAPSIZE'])
                if score >= int(game_settings['MAXPOINTS']):
                    playery = -100
                show_score = game_settings.getboolean('SHOWSCORE')

                # with open('settings.txt', 'r') as infile:
                #     FPS = int(infile.readline().strip())
                break
            except Exception as e:
                print(e)
                continue

        FPSCLOCK.tick(FPS)


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe(pipegapsize=PIPEGAPSIZE):
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - pipegapsize))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 35
    # print('gap',gapY)
    # print('ph',pipeHeight)
    # print()

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + pipegapsize}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()

def showData(data):
    y = 430
    x = 10
    i = 0
    for k, v in data.items():
        text=f'{k}: {v}'
        draw_text(SCREEN, text, size=12,
                  color=(0, 0, 0), x=x, y=y)
        i += 1
        y += 20
        if i%4 == 0:
            x += 100
            y = 430

def draw_text(surface, text, size, color, x, y, align='nw'):
    font_name = pygame.font.match_font('arial')
    font = pygame.font.Font(font_name, size)
    text_surf = font.render(str(text), True, color)
    text_rect = text_surf.get_rect()
    if align == 'nw':
        text_rect.topleft = (x, y)
    if align == 'ne':
        text_rect.topright = (x, y)
    if align == 'sw':
        text_rect.bottomleft = (x, y)
    if align == 'se':
        text_rect.bottomright = (x, y)
    if align == 'n':
        text_rect.midtop = (x, y)
    if align == 's':
        text_rect.midbottom = (x, y)
    if align == 'e':
        text_rect.midright = (x, y)
    if align == 'w':
        text_rect.midleft = (x, y)
    if align == 'center':
        text_rect.center = (x, y)
    surface.blit(text_surf, text_rect)


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    elif player['y'] < -10:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main(network=None)
