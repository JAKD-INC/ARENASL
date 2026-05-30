import { expect, test } from 'vitest'
import { splitHands } from './landmarks.ts'

const L = [[0.1, 0, 0]]
const R = [[0.9, 0, 0]]

test('assigns hands by handedness label', () => {
  const { handLeft, handRight } = splitHands(
    [L, R],
    [[{ categoryName: 'Left' }], [{ categoryName: 'Right' }]],
  )
  expect(handLeft).toEqual(L)
  expect(handRight).toEqual(R)
})

test('missing hand stays null', () => {
  const { handLeft, handRight } = splitHands([R], [[{ categoryName: 'Right' }]])
  expect(handLeft).toBeNull()
  expect(handRight).toEqual(R)
})
