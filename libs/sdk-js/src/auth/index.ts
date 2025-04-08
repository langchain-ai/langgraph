import type {
  AuthenticateCallback,
  AnyCallback,
  CallbackEvent,
  OnCallback,
  BaseAuthReturn,
  ToUserLike,
  BaseUser,
} from "./types.js";

export class Auth<
  TExtra,
  TAuthReturn extends BaseAuthReturn = BaseAuthReturn,
  TUser extends BaseUser = ToUserLike<TAuthReturn>,
> {
  "~handlerCache": {
    authenticate?: AuthenticateCallback<BaseAuthReturn>;
    callbacks?: Record<string, AnyCallback>;
  } = {};

  authenticate<T extends BaseAuthReturn>(
    cb: AuthenticateCallback<T>,
  ): Auth<TExtra, T> {
    this["~handlerCache"].authenticate = cb;
    return this as unknown as Auth<TExtra, T>;
  }

  on<T extends CallbackEvent>(event: T, callback: OnCallback<T, TUser>): this {
    this["~handlerCache"].callbacks ??= {};
    this["~handlerCache"].callbacks[event as string] = callback as AnyCallback;
    return this;
  }
}

export type { Filters, ResourceActionType } from "./types.js";
export { HTTPException } from "./error.js";
