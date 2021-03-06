from itertools import cycle
import configparser
import random
import sys

import time
import pygame
from pygame.locals import *


# MOST OF THIS CODE TAKEN DIRECTLY FROM
# https://github.com/sourabhv/FlapPyBird
# The MIT License (MIT)
#
# Copyright (c) <year> <copyright holders>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.



FPS = 200
SCREENWIDTH  = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79

PARSER = configparser.ConfigParser()
PARSER.read('configs.txt')
game_settings = PARSER['GAME']
if 'FPS' in game_settings:
    FPS = int(game_settings['FPS'])
if 'PIPEGAPSIZE' in game_settings:
    PIPEGAPSIZE = int(game_settings['PIPEGAPSIZE'])


pygame.init()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))


# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

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


def main(network=None, pipegapsizes=None):
    """
    Main function can accept a Neat Network object and determine how
    well the network can play flappy bird.

    Parameters
    ----------
    network (neat network object): a network with 8 inputs and one output

    Returns
    -------
    score

    """
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))





    pygame.display.set_caption('Flappy Bird')


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

        # hitmask for pipes
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
        crashInfo = mainGame(network=network, pipegapsizes=pipegapsizes)

        # Get total time
        totTime = pygame.time.get_ticks()
        #print(crashInfo['score'],  end='\t')

        return crashInfo['score']


def mainGame(network=None, pipegapsizes=None):
    i_pipegap = 0
    if pipegapsizes is not None:
        #print(pipegapsizes)
        pipegapsize = pipegapsizes[i_pipegap]


    score = playerIndex = loopIter = 0
    while True:
        try:
            PARSER.read('configs.txt')
            game_settings = PARSER['GAME']
            FPS = int(game_settings['FPS'])
            if game_settings.getboolean('USETHISPIPEGAP'):
                pipegapsize = int(game_settings['PIPEGAPSIZE'])
            if score >= int(game_settings['MAXPOINTS']):
                playery = -100
            is_display = game_settings.getboolean('ISDISPLAY')
            is_show_score = game_settings.getboolean('ISSHOWSCORE') & is_display
            is_sound = game_settings.getboolean('ISSOUND')
            flip_colors = game_settings.getboolean('FLIPCOLORS')
            break
        except Exception as e:
            print(e)
            continue


    movementInfo = {
                    'playery': int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2),
                    'basex': 0,
                    'playerIndexGen': cycle([0, 1, 2, 1]),
                }

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
        # Update settings
        while True:
            try:
                PARSER.read('configs.txt')
                game_settings = PARSER['GAME']
                FPS = int(game_settings['FPS'])
                if game_settings.getboolean('USETHISPIPEGAP'):
                    pipegapsize = int(game_settings['PIPEGAPSIZE'])
                if score >= int(game_settings['MAXPOINTS']):
                    score *= 2
                    playery = -100
                is_display = game_settings.getboolean('ISDISPLAY')
                is_show_score = game_settings.getboolean('ISSHOWSCORE') & is_display
                is_sound = game_settings.getboolean('ISSOUND')
                flip_colors = game_settings.getboolean('FLIPCOLORS')

                # with open('settings.txt', 'r') as infile:
                #     FPS = int(infile.readline().strip())
                break
            except Exception as e:
                print(e)
                continue

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
        pipe_w = IMAGES['pipe'][0].get_width()
        pipe_h = IMAGES['pipe'][0].get_height()
        for i in range(len(upperPipes)):
            if playerMidPos < upperPipes[i]['x'] + pipe_w:
                # next two pipe x pos
                uPipe1X = upperPipes[i]['x']
                uPipe2X = upperPipes[i+1]['x']

                # next two pipe y pos
                uPipe1Y = upperPipes[i]['y'] + pipe_h
                uPipe2Y = upperPipes[i+1]['y'] + pipe_h

                break

        for i in range(len(lowerPipes)):
            if playerMidPos < lowerPipes[i]['x'] + pipe_w:
                # next two pipe x pos
                lPipe1X = lowerPipes[i]['x']
                lPipe2X = lowerPipes[i+1]['x']

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
                # round up to nearest 100 when pass pipe (awarding 62 or 64
                # points)
                #score += 50
                score = int((score // 100 + 1))*100
                if is_sound:
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
            if pipegapsizes is not None:
                i_pipegap = min(i_pipegap + 1, len(pipegapsizes)-1)
                pipegapsize = pipegapsizes[i_pipegap]
            if game_settings.getboolean('USETHISPIPEGAP'):
                pipegapsize = int(game_settings['PIPEGAPSIZE'])
            newPipe = getRandomPipe(pipegapsize)
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # Player rotation has a threshold
        visibleRot = playerRotThr
        if playerRot <= playerRotThr:
            visibleRot = playerRot

        if is_display:
            # draw sprites
            SCREEN.blit(IMAGES['background'], (0,0))

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

            # For visual effect wanted for recording
            if False:
                close = 200
                if 880 < score < 967:
                    uPipe = upperPipes[1]
                    lPipe = lowerPipes[1]
                    SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']+close))
                    SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']-close))

                if 967 <= score < 1055:
                    uPipe = upperPipes[0]
                    lPipe = lowerPipes[0]
                    SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']+close))
                    SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']-close))

            SCREEN.blit(IMAGES['base'], (basex, BASEY))
            # print score so player overlaps the score


            playerSurface = pygame.transform.rotate(IMAGES['player'][playerIndex], visibleRot)
            SCREEN.blit(playerSurface, (playerx, playery))

            if flip_colors:
                pixels = pygame.surfarray.pixels2d(SCREEN)
                pixels ^= 2**32 - 1
                del pixels
            #pygame.display.flip()

        if is_show_score:
            showScore(score)
            showData(input_dict)

        if is_display:
            pygame.display.update()

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
    #gapY = random.randrange(0, int(BASEY * 0.6 - pipegapsize))
    gapY = random.randrange(0, int(BASEY * 0.7 - pipegapsize))
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
                #print(x1+x, y1+y, x2+x, y2+y)
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

def pipeDisplay():
    """Build the display for the powerpoint presentation"""
    pygame.init()
    SCREENWIDTH = 300
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    FPSCLOCK = pygame.time.Clock()

    pipeindex=0
    randBg=0
    basex = 0

    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )
    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

    pipeX = SCREENWIDTH + 35

    for i in range(10):
        SCREEN.blit(IMAGES['background'], (i*255,0))

    n_pipes=1
    for i in range(1,n_pipes+1):
        pipeX = i*38*4
        pipegapsize = 90
        gapY = random.randrange(0, int(BASEY * 0.7 - pipegapsize))
        gapY += int(BASEY * 0.2)
        pipeHeight = IMAGES['pipe'][0].get_height()
        upperPipes = [
            {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        ]
        lowerPipes = [
            {'x': pipeX, 'y': gapY + pipegapsize}, # lower pipe
        ]
        upperPipes = [
            {'x': 152, 'y': -199},  # upper pipe
        ]
        lowerPipes = [
            {'x': 152, 'y': 211}, # lower pipe
        ]
        print(upperPipes)
        print(lowerPipes)

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

    for i, (x, y) in enumerate([(120, 50), (40, 380), (160, 160)]):
        print(i)
        randPlayer = i
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
            )
        playerx = 50
        playery = 50
        SCREEN.blit(IMAGES['player'][0], (x, y))

    for i in range(10):
        SCREEN.blit(IMAGES['base'], (i*255, BASEY))
    pygame.display.update()

if __name__ == '__main__':
   # main(network=None)
    main(network=None, pipegapsizes=[180, 170, 160, 150, 140, 130, 100, 80])
